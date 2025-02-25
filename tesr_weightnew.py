import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

# 3D CNN 模型
class PigAction3DCNN(nn.Module):
    def __init__(self, num_classes=2):  # 二分類
        super(PigAction3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        # 假設輸入最終在 conv2 + pool2 後會變成 (batch_size, 64, 1, 56, 56)
        # 如果 sequence_length=5, 224x224影像 => shape 需確保最終 Flatten 大小一致
        # 這裡寫死 64 * 1 * 56 * 56，可根據實際 shape 做更保險的實作
        self.fc1 = nn.Linear(64 * 1 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 3, sequence_length, 224, 224)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # x shape => (batch_size, 64, 1, 56, 56) if sequence_length=5, image=224x224
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 自定義 Dataset (改成讀取 CSV、.jpg)
class PigDataset(Dataset):
    def __init__(self, image_folder, label_file, sequence_length=5, transform=None):
        self.image_folder = image_folder
        # 改用 pd.read_csv()
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        expected_columns = ['frame_file', 'action']
        if not all(col in self.labels.columns for col in expected_columns):
            raise ValueError(f"列名不匹配，當前列名為：{self.labels.columns}")

        # 二分類：normal / transition
        self.label_map = {
            "normal": 0,
            "transition": 1
        }
        self.labels['label'] = self.labels['action'].map(self.label_map)
        
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.labels) - self.sequence_length + 1

    def __getitem__(self, idx):
        images = []
        for i in range(self.sequence_length):
            img_name = self.labels.iloc[idx + i]['frame_file']  # 假設這裡是 xxx.jpg
            img_path = os.path.join(self.image_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # shape: (sequence_length, C, H, W)
        images = torch.stack(images)
        label = int(self.labels.iloc[idx + self.sequence_length - 1]['label'])
        return images, label

def get_test_loader(image_folder, label_file, batch_size=16, sequence_length=5):
    dataset = PigDataset(image_folder, label_file, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def test_model(model_path, test_loader, device):
    model = PigAction3DCNN(num_classes=2)  # 二分類模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # (batch_size, sequence_length, C, H, W) -> (batch_size, C, sequence_length, H, W)
            images = images.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.2f}%")
    return np.array(all_predictions), np.array(all_labels), accuracy

def evaluate_results(predictions, labels, label_map):
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
    label_names = {v: k for k, v in label_map.items()}

    conf_matrix = confusion_matrix(labels, predictions, labels=list(label_map.values()))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # precision / recall 各類別
    precision = precision_score(labels, predictions, labels=list(label_map.values()), average=None)
    recall = recall_score(labels, predictions, labels=list(label_map.values()), average=None)
    print("\nPrecision and Recall by Class:")
    for idx, label in enumerate(label_map.keys()):
        print(f"{label}: Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}")

    # 分類報告
    report = classification_report(labels, predictions, target_names=label_names.values(), digits=4)
    print("\nClassification Report:\n", report)

def save_results(predictions, labels, label_map, output_file="test_results.csv"):
    label_names = {v: k for k, v in label_map.items()}
    pred_names = [label_names.get(p, "unknown") for p in predictions]
    true_names = [label_names.get(l, "unknown") for l in labels]

    results = pd.DataFrame({
        "True Label": true_names,
        "Predicted Label": pred_names
    })
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"測試結果已儲存至 {output_file}")

if __name__ == "__main__":
    # 測試設定 (請自行修改路徑及檔名)
    test_images_path = "/home/yuchu/pig/dataset/val"   # 存放 .jpg 檔案的資料夾
    test_labels_file = "/home/yuchu/pig/dataset/val.csv"          # CSV 檔案
    model_path = "/home/yuchu/pig/bbbest_3dcnn_binary_model.pth"
    batch_size = 64
    sequence_length = 5

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載測試數據
    test_loader = get_test_loader(test_images_path, test_labels_file, batch_size=batch_size, sequence_length=sequence_length)

    # 測試模型
    label_map = {
        "normal": 0,
        "transition": 1
    }
    predictions, labels, accuracy = test_model(model_path, test_loader, device)

    # 評估結果
    evaluate_results(predictions, labels, label_map)

    # 儲存測試結果
    save_results(predictions, labels, label_map, output_file="test_results.csv")
