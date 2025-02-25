import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 自定義 Dataset，改用讀取 .csv 檔
class PigDataset(Dataset):
    def __init__(self, image_folder, label_file, sequence_length=5, transform=None):
        self.image_folder = image_folder
        # 改用 pd.read_csv() 讀取標籤
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()
        
        # 假設 csv 檔仍然有 'time' 與 'action' 兩欄
        expected_columns = ['frame_file', 'action']
        if not all(col in self.labels.columns for col in expected_columns):
            raise ValueError(f"列名不匹配，當前列名為：{self.labels.columns}")
        
        # 修改類別映射為兩類
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
            img_name = self.labels.iloc[idx + i]['frame_file']  
            # 假設 csv 內的 'time' 欄位已經包含檔名（例如 XXX.jpg）
            img_path = os.path.join(self.image_folder, img_name)
            
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images)  # shape: (sequence_length, C, H, W)
        label = int(self.labels.iloc[idx + self.sequence_length - 1]['label'])
        
        return images, label

def get_dataloader(image_folder, label_file, batch_size=16, sequence_length=5, shuffle=True):
    dataset = PigDataset(image_folder, label_file, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 3D CNN 模型 (二分類)
class PigAction3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PigAction3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, C, sequence_length, H, W)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # x shape 可能變化，取決於 sequence_length、影像大小等
        
        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2) * x.size(3) * x.size(4)
            self.fc1 = nn.Linear(flattened_size, 128).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ======== 以下為訓練流程範例 ========

# 假設您已將訓練影像存放在 /home/yuchu/pig/train_frames_jpg/，檔名都為 .jpg
# 並且標籤為 /home/yuchu/pig/train.csv，內含 'time' 與 'action' 欄位
train_loader = get_dataloader(
    image_folder="/home/yuchu/pig/dataset/train12/",
    label_file="/home/yuchu/pig/dataset/train12/train1.csv",
    batch_size=64,
    sequence_length=5,
    shuffle=True
)
val_loader = get_dataloader(
    image_folder="/home/yuchu/pig/dataset/train3_test1/",
    label_file="/home/yuchu/pig/dataset/train3_test1/train3_test1.csv",
    batch_size=64,
    sequence_length=5,
    shuffle=False
)

model = PigAction3DCNN(num_classes=2)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 假設我們已有類別計數
class_counts = {
    "normal": 3858,
    "transition": 1698
}
total_samples = sum(class_counts.values())
weights = torch.tensor(
    [total_samples / class_counts[action] for action in ["normal", "transition"]],
    dtype=torch.float32
).to(device)

print(f"Class weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
best_val_accuracy = 0.0
best_model_path = "best_3dcnn_binary_model.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        # 原先您在訓練時會把維度調成 (batch_size, C, seq_len, H, W)
        # images.shape: (batch_size, seq_len, C, H, W)
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

    # 驗證階段
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
    
    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

print("訓練完成，最佳模型已保存！")
