import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.amp import autocast

# 自定義 Focal Loss
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C)
        # targets: (N,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PigNPZDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.image_sequences = []
        self.labels = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        print("Processing NPZ files...")
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
        # (sequence_length, C, H, W)
        images = self.image_sequences[idx]
        processed_images = []

        for time_step in images:
            # (C, H, W) -> (H, W, C)
            img = np.transpose(time_step, (1, 2, 0))

            # 若原本為float且在[0,1]，轉為uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            pil_img = Image.fromarray(img).convert('RGB')
            processed_images.append(self.transform(pil_img))

        # 堆疊為 (sequence_length, C, H, W)
        processed_images = torch.stack(processed_images, dim=0)
        # 模型期望 (C, T, H, W)
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
        y = x.view(b, c, -1).mean(dim=2)  # Global average pooling over t,h,w
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


# Paths to NPZ files
train_files = [
    #"/home/yuchu/pig/dataset/seg3.npz",
    #"/home/yuchu/pig/dataset/seg4.npz",
    "/home/yuchu/pig/dataset/seg5.npz"
    #"/home/yuchu/pig/dataset/seg6.npz"
]
val_files = ["/home/yuchu/pig/dataset/seg2.npz"]

# Create DataLoaders
sequence_length = 5  # 若您需要用在 compute_class_weights 時
train_loader = DataLoader(PigNPZDataset(train_files), batch_size=16, shuffle=True)
val_loader = DataLoader(PigNPZDataset(val_files), batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = PigAction3DCNN(num_classes=2).to(device)

# Initialize flattened_size using dummy input
num_images_per_time_step = 1  
temporal_dimension = sequence_length * num_images_per_time_step
dummy_input = torch.zeros(1, 3, temporal_dimension, 224, 224).to(device)
with torch.no_grad():
    model(dummy_input)  

def compute_class_weights(file_paths, sequence_length):
    labels = []
    for file_path in file_paths:
        data = np.load(file_path)
        labels.extend(data['labels'][sequence_length - 1:])
    class_counts = pd.Series(labels).value_counts().to_dict()
    total_samples = sum(class_counts.values())
    weights = torch.tensor(
        [total_samples / class_counts.get(cls, 1) for cls in range(2)],
        dtype=torch.float32
    )
    return weights

# 如果您要考慮class_weights可在此查看，但使用FocalLoss時可先不使用此參數
# weights = compute_class_weights(train_files, sequence_length=sequence_length).to(device)

# 使用FocalLoss，根據需要調整alpha與gamma
# 例如alpha=2.0表示給相對少數的class更多的影響力
# gamma=2.0是常見設定，可嘗試1.5、3.0等不同值
criterion = FocalLoss(alpha=2.0, gamma=2.0, reduction='mean').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 50
best_val_loss = float('inf')
best_model_path = "best_3dcnn_model.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, total_samples = 0.0, 0, 0
    print(f"Epoch {epoch+1}/{num_epochs}: Training...")
    scaler = torch.cuda.amp.GradScaler()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = train_correct / total_samples * 100
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation phase
    model.eval()
    val_loss, val_correct, val_samples = 0.0, 0, 0
    print(f"Epoch {epoch+1}/{num_epochs}: Validating...")
    for images, labels in tqdm(val_loader, desc="Validation Progress", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_samples += labels.size(0)

    val_accuracy = val_correct / val_samples * 100
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'flattened_size': model.flattened_size,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
    scheduler.step(val_loss)
