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
# Dataset：從.jpg中讀取序列影像
# -------------------------
class PigDataset(Dataset):
    def __init__(self, img_folder, label_file, sequence_length=15, device='cpu'):
        self.img_folder = img_folder
        self.sequence_length = sequence_length
        self.device = device

        # 讀取標籤檔
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        if 'action' not in self.labels.columns or 'frame_file' not in self.labels.columns:
            raise ValueError("CSV必須包含 'action' 與 'filename' 欄位")

        self.labels['label'] = self.labels['action'].map({
            "normal": 0,
            "transition": 1
        })
        if 'label' not in self.labels.columns:
            raise ValueError("'label' 欄位生成失敗，請檢查 'action' 欄位")

        self.filenames = self.labels['frame_file'].values

        # 影像轉換
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
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

def get_dataloader(img_folder, label_file, batch_size=16, sequence_length=15, shuffle=False, device='cpu'):
    dataset = PigDataset(img_folder, label_file, sequence_length=sequence_length, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# -------------------------
# 模型定義 (與訓練時相同)
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
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

    def forward(self, x):
        # x: (batch_size * seq_len, 3, H, W)
        x = self.cnn(x)  # (batch_size*seq_len, 512, 1, 1)
        x = x.view(x.size(0), -1) # (batch_size*seq_len, 512)
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
        output = self.fc(x) # (batch_size, num_classes)
        return output

class VideoClassifier(nn.Module):
    def __init__(self, cnn_extractor, transformer_model):
        super(VideoClassifier, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.transformer_model = transformer_model

    def forward(self, x):
        # x: (batch_size, seq_len, 3, H, W)
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size*seq_length, c, h, w)
        features = self.cnn_extractor(x)  # (batch_size*seq_len, 512)
        features = features.view(seq_length, batch_size, -1) # (seq_len, batch_size, 512)
        output = self.transformer_model(features)
        return output

# -------------------------
# 測試程式碼 (Inference)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_img_folder = "/home/yuchu/pig/dataset/test"  # 請換成您的test影像資料夾
test_csv = "/home/yuchu/pig/dataset/test.csv"

batch_size = 16
sequence_length = 15
num_classes = 2
input_size = 512
num_heads = 4
num_layers = 2

test_loader = get_dataloader(
    img_folder=test_img_folder,
    label_file=test_csv,
    batch_size=batch_size,
    sequence_length=sequence_length,
    shuffle=False,
    device=device
)

cnn_extractor = CNNFeatureExtractor()
transformer_model = TransformerClassifier(input_size=input_size, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
model = VideoClassifier(cnn_extractor, transformer_model).to(device)

model_path = "model.pth"  # 請確定此路徑與檔名
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
test_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

avg_test_loss = test_loss / total_samples
test_accuracy = 100 * test_correct / total_samples

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
