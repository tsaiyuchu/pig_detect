import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# -------------------------
# Dataset 定義：改為從硬碟讀取 .jpg 圖片
# -------------------------
class PigDataset(Dataset):
    def __init__(self, img_folder, label_file, sequence_length=15, device='cpu'):
        self.img_folder = img_folder
        self.sequence_length = sequence_length
        self.device = device

        # 讀取標籤文件
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        # 檢查 'action' 列並映射到 'label'
        if 'action' in self.labels.columns and 'frame_file' in self.labels.columns:
            self.labels['label'] = self.labels['action'].map({
                "normal": 0,
                "transition": 1
            })
        else:
            raise ValueError("CSV 文件中必須包含 'action' 與 'filename' 欄位")

        # 確認 'label' 列是否存在
        if 'label' not in self.labels.columns:
            raise ValueError("'label' 列未成功生成，請檢查 'action' 列和映射邏輯")

        # 取得所有的檔名（確保其排序為時間序列順序）
        self.filenames = self.labels['frame_file'].values

        # 影像轉換 (可根據需要調整)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames) - self.sequence_length + 1

    def __getitem__(self, idx):
        # 取得 sequence_length 張連續影像
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

# -------------------------
# 模型定義 (與之前相同)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = models.resnet18(weights='DEFAULT')
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 移除最後全連接層

    def forward(self, x):
        # x: (batch_size * seq_len, 3, H, W)
        x = self.cnn(x)  # (batch_size*seq_len, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size*seq_len, 512)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, num_classes, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.positional_encoding = PositionalEncoding(input_size, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        # x: (seq_len, batch_size, input_size)
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros((x.size(1), x.size(0)), dtype=torch.bool, device=x.device)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=0)  # (batch_size, input_size)
        output = self.fc(x)  # (batch_size, num_classes)
        return output

class VideoClassifier(nn.Module):
    def __init__(self, cnn_extractor, transformer_model):
        super(VideoClassifier, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.transformer_model = transformer_model

    def forward(self, x):
        # x: (batch_size, seq_len, C, H, W)
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.cnn_extractor(x)  # (batch_size*seq_len, 512)
        features = features.view(seq_length, batch_size, -1)  # (seq_len, batch_size, 512)
        output = self.transformer_model(features)
        return output

# -------------------------
# 訓練與驗證流程設定
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定資料集路徑 (假設您的影像都放在 /path/to/images/ 中)
train_img_folder = "/home/yuchu/pig/dataset/train/"
train_csv = "/home/yuchu/pig/dataset/train.csv"
val_img_folder = "/home/yuchu/pig/dataset/val/"
val_csv = "/home/yuchu/pig/dataset/val.csv"

batch_size = 16
sequence_length = 15
num_classes = 2
learning_rate = 0.001
num_epochs = 100
input_size = 512  # ResNet18的最後輸出維度
num_heads = 4
num_layers = 2

train_loader = get_dataloader(
    img_folder=train_img_folder,
    label_file=train_csv,
    batch_size=batch_size,
    sequence_length=sequence_length,
    shuffle=True,
    device=device
)

val_loader = get_dataloader(
    img_folder=val_img_folder,
    label_file=val_csv,
    batch_size=batch_size,
    sequence_length=sequence_length,
    shuffle=False,
    device=device
)

cnn_extractor = CNNFeatureExtractor()
transformer_model = TransformerClassifier(input_size=input_size, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
model = VideoClassifier(cnn_extractor, transformer_model).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_val_accuracy = 0.0
save_path = "model.pth"

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)  # (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = 100 * train_correct / total_samples
    avg_train_loss = train_loss / total_samples

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item() * val_labels.size(0)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / val_total

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # 若當前驗證準確率高於歷史最佳，儲存模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with Val Accuracy: {best_val_accuracy:.2f}%")

print("訓練完成！最佳模型已保存於", save_path)
