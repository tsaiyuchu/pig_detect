import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class PigDataset(Dataset):
    def __init__(self, img_folder, label_file, sequence_length=15, device='cpu'):
        self.img_folder = img_folder
        self.sequence_length = sequence_length
        self.device = device

        # 讀取標籤文件
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        # 確保有 'action' 與 'filename' 欄位
        if 'action' not in self.labels.columns or 'frame_file' not in self.labels.columns:
            raise ValueError("CSV 文件必須包含 'action' 與 'filename' 欄位")

        # 將 action 映射成 label
        self.labels['label'] = self.labels['action'].map({
            "normal": 0,
            "transition": 1
        })

        if 'label' not in self.labels.columns:
            raise ValueError("'label' 欄位生成失敗，請檢查 'action' 欄位")

        self.filenames = self.labels['frame_file'].values

        # 定義影像轉換
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
            img = self.transform(img)  # (C, H, W)
            images.append(img)
        images = torch.stack(images, dim=0).to(self.device)  # (sequence_length, C, H, W)

        label = torch.tensor(self.labels.iloc[idx+self.sequence_length-1]['label'], dtype=torch.long).to(self.device)
        return images, label

def get_dataloader(img_folder, label_file, batch_size=16, sequence_length=15, shuffle=True, device='cpu'):
    dataset = PigDataset(img_folder, label_file, sequence_length=sequence_length, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 3D CNN + LSTM 模型 (與之前相同)
class PigAction3DCNNLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=128, lstm_num_layers=1, num_classes=2):
        super(PigAction3DCNNLSTM, self).__init__()
        # 卷積與池化層結構
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes

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

# 訓練與驗證流程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_folder = "/home/yuchu/pig/dataset/train/"  # 請換成您的train影像資料夾路徑
train_csv = "/home/yuchu/pig/dataset/train.csv"

val_img_folder = "/home/yuchu/pig/dataset/val/"    # 請換成您的val影像資料夾路徑
val_csv = "/home/yuchu/pig/dataset/val.csv"

train_loader = get_dataloader(
    img_folder=train_img_folder,
    label_file=train_csv,
    batch_size=16,
    sequence_length=15,
    shuffle=True,
    device=device
)

val_loader = get_dataloader(
    img_folder=val_img_folder,
    label_file=val_csv,
    batch_size=16,
    sequence_length=15,
    shuffle=False,
    device=device
)

class_counts = {
    "normal": 4877,
    "transition": 2185
}
total_samples = sum(class_counts.values())
weights = torch.tensor(
    [total_samples / class_counts[action] for action in ["normal", "transition"]],
    dtype=torch.float32
).to(device)

model = PigAction3DCNNLSTM(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
best_val_accuracy = 0.0
best_model_path = "3dcnn_lstm_model.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = 100 * train_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    val_accuracy = 100 * val_correct / total_samples
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

print("訓練完成，最佳模型已保存！")
