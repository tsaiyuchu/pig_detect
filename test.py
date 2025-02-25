import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class PigNPZDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.image_sequences = []
        self.labels = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        print("Processing NPZ files for testing...")
        for file_path in tqdm(file_paths, desc="NPZ Files"):
            data = np.load(file_path)
            # images shape: (num_samples, sequence_length, C, H, W)
            images = data['images']
            labels = data['labels']  # (num_samples,)

            for seq_imgs, lbl in zip(images, labels):
                self.image_sequences.append(seq_imgs)
                self.labels.append(lbl)

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        images = self.image_sequences[idx]  # (sequence_length, C, H, W)
        processed_images = []

        for time_step in images:
            img = np.transpose(time_step, (1, 2, 0))  # (C,H,W)->(H,W,C)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img).convert('RGB')
            processed_images.append(self.transform(pil_img))

        # 堆疊為 (sequence_length, C, H, W)
        processed_images = torch.stack(processed_images, dim=0)
        # 模型期望輸入 (C, T, H, W)
        processed_images = processed_images.permute(1, 0, 2, 3)

        label = self.labels[idx]
        return processed_images, label


# Squeeze and Excitation Block
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y


# 3D CNN model
class PigAction3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PigAction3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SqueezeExcitationBlock(64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SqueezeExcitationBlock(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SqueezeExcitationBlock(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.flattened_size = None
        self.classifier = None

    def forward(self, x):
        x = self.features(x)
        if self.flattened_size is None:
            self.flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Linear(self.flattened_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 2)
            ).to(x.device)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 測試檔案路徑
test_files = ["/home/yuchu/pig/dataset/seg1.npz"]

test_loader = DataLoader(PigNPZDataset(test_files), batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = PigAction3DCNN(num_classes=2).to(device)

# 因為model在forward時初始化classifier，所以需要一個dummy_input來確定flattened_size
sequence_length = 5
num_images_per_time_step = 1
temporal_dimension = sequence_length * num_images_per_time_step
dummy_input = torch.zeros(1, 3, temporal_dimension, 224, 224).to(device)
with torch.no_grad():
    model(dummy_input)

# 載入訓練好的權重
checkpoint = torch.load("best_3dcnn_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
test_correct = 0
test_total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 若需要顯示詳細預測結果
print("Predictions:", all_preds)
print("True Labels:", all_labels)
