import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PigDataset(Dataset):
    def __init__(self, image_folder, label_file, sequence_length=5, transform=None):
        self.image_folder = image_folder
        self.labels = pd.read_csv(label_file)  # 假設 CSV 格式為: [filename, label]
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.labels) - self.sequence_length + 1  # 可生成的序列數量

    def __getitem__(self, idx):
        images = []
        for i in range(self.sequence_length):
            img_name = self.labels.iloc[idx + i, 0]
            img_path = os.path.join(self.image_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)  # (sequence_length, C, H, W)
        label = self.labels.iloc[idx + self.sequence_length - 1, 1]  # 標籤為序列末幀的動作
        return images, label

# 創建數據集和 DataLoader
train_dataset = PigDataset("train_images/", "train_labels.csv", sequence_length=5)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = PigDataset("val_images/", "val_labels.csv", sequence_length=5)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
