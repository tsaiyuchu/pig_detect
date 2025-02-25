import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Dataset: 從 JPG 圖片讀取序列影像
class PigDataset(Dataset):
    def __init__(self, img_folder, label_file, sequence_length=15, device='cpu'):
        self.img_folder = img_folder
        self.sequence_length = sequence_length
        self.device = device

        # 讀取標籤文件
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        if 'action' not in self.labels.columns or 'frame_file' not in self.labels.columns:
            raise ValueError("CSV必須包含 'action' 與 'frame_file' 欄位")

        self.labels['label'] = self.labels['action'].map({
            "normal": 0,
            "transition": 1
        })

        if 'label' not in self.labels.columns:
            raise ValueError("'label' 欄位生成失敗，請檢查 'action' 欄位")

        self.filenames = self.labels['frame_file'].values

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq_filenames = self.filenames[idx:idx+self.sequence_length]
        images = []
        for fname in seq_filenames:
            img_path = os.path.join(self.img_folder, fname)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images, dim=0).to(self.device)  # (sequence_length, C, H, W)

        label = torch.tensor(self.labels.iloc[idx+self.sequence_length-1]['label'], dtype=torch.long).to(self.device)
        return images, label

# DataLoader
def get_dataloader(img_folder, label_file, batch_size=16, sequence_length=15, shuffle=False, device='cpu'):
    dataset = PigDataset(img_folder, label_file, sequence_length=sequence_length, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 定義 PigAction3DCNNLSTM 模型
class PigAction3DCNNLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=128, lstm_num_layers=1, num_classes=2):
        super(PigAction3DCNNLSTM, self).__init__()
        # 卷積與池化層結構
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256 * 28 * 28,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)

        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)

        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.pool3(x)

        batch_size, channels, depth, height, width = x.size()
        x = x.contiguous().view(batch_size, depth, -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入訓練好的模型
model = PigAction3DCNNLSTM(num_classes=2).to(device)
model.load_state_dict(torch.load("/home/yuchu/pig/3dcnn_lstm_model.pth", map_location=device))
model.eval()

# 設定測試資料位置與載入
test_img_folder = "/home/yuchu/pig/dataset/test"  # 請更新為您的實際路徑
test_csv = "/home/yuchu/pig/dataset/test.csv"
test_loader = get_dataloader(
    img_folder=test_img_folder,
    label_file=test_csv,
    batch_size=16,
    sequence_length=15,
    shuffle=False,
    device=device
)

# 評估模型
correct = 0
total = 0
class_correct = [0, 0]
class_total = [0, 0]
results = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            class_total[label] += 1
            class_correct[label] += (pred == label)

            # 注意: test_loader.dataset.filenames 是按照 dataset 定義，我們需取得對應的影像名稱
            # 因為我們在 __getitem__ 中使用 idx 取得 seq，因此需要計算出對應的文件名
            # idx = batch_idx * batch_size + i 是此樣本在整個 dataset 的序號
            global_idx = batch_idx * test_loader.batch_size + i
            filename_idx = global_idx  # filename 對應需要和 dataset 中的 idx 對應
            # filename_idx 對於序列資料，對第 (filename_idx) 筆資料對應的實際filename：
            # dataset中 __getitem__ 回傳的是 idx: idx+self.sequence_length 的序列
            # 但我們列印 label 時是從最後一張 idx+self.sequence_length-1 取得
            # 因此對於該序列的 filename 應該是 dataset.filenames[filename_idx+self.sequence_length-1]
            
            sequence_start = filename_idx
            filename_for_this_sample = test_loader.dataset.filenames[sequence_start + test_loader.dataset.sequence_length - 1]

            results.append({
                "filename": filename_for_this_sample,
                "true_label": "normal" if label == 0 else "transition",
                "predicted_label": "normal" if pred == 0 else "transition"
            })

overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

for i, cls in enumerate(["normal", "transition"]):
    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Class '{cls}' Accuracy: {class_accuracy:.2f}%")

# 儲存結果至 CSV
output_csv_path = "test_predictions.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")
