import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 自定義 Dataset
class PigDataset(Dataset):
    def __init__(self, image_folder, label_file, sequence_length=15, transform=None):
        self.image_folder = image_folder
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()
        
        # 檢查列名
        expected_columns = ['frame_file', 'action']
        if not all(col in self.labels.columns for col in expected_columns):
            raise ValueError(f"列名不匹配，當前列名為：{self.labels.columns}")
        
        # 類別映射
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
            img_path = os.path.join(self.image_folder, img_name)

            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)  # shape: (sequence_length, C, H, W)
        label = int(self.labels.iloc[idx + self.sequence_length - 1]['label'])

        return images, label

def get_dataloader(image_folder, label_file, batch_size=16, sequence_length=15, shuffle=True):
    dataset = PigDataset(image_folder, label_file, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 3D CNN + LSTM 模型
class PigAction3DCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=128, lstm_num_layers=1):
        super(PigAction3DCNNLSTM, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.feature_size = None
        self.lstm = None

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        if self.feature_size is None:
            with torch.no_grad():
                sample_input = torch.zeros(1, x.size(1), x.size(2), x.size(3), x.size(4)).to(x.device)
                sample_output = self.pool2(torch.relu(self.conv2(self.pool1(torch.relu(self.conv1(sample_input))))))
                _, channels, _, height, width = sample_output.size()
                self.feature_size = channels * height * width
                self.lstm = nn.LSTM(
                    input_size=self.feature_size,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True
                ).to(x.device)

        # x shape: (batch_size, C, sequence_length, H, W)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        
        # x shape: (batch_size, 64, seq_len, H, W)
        batch_size, channels, seq_len, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # shape: (batch_size, seq_len, channels, H, W)
        x = x.reshape(batch_size, seq_len, -1)  # shape: (batch_size, seq_len, self.feature_size)

        # LSTM
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, lstm_hidden_size)
        
        # 取最後一個時間步的輸出
        x = lstm_out[:, -1, :]  # shape: (batch_size, lstm_hidden_size)

        # 全連接層
        x = self.fc(x)  # shape: (batch_size, num_classes)
        return x

# 訓練流程類似於原始程式
train_loader = get_dataloader(
    image_folder="/home/yuchu/pig/dataset/train12/",
    label_file="/home/yuchu/pig/dataset/train12/train12.csv",
    batch_size=16,
    sequence_length=15,
    shuffle=True
)
val_loader = get_dataloader(
    image_folder="/home/yuchu/pig/dataset/val1/",
    label_file="/home/yuchu/pig/dataset/val1/val.csv",
    batch_size=16,
    sequence_length=15,
    shuffle=False
)

model = PigAction3DCNNLSTM(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用類別權重
class_counts = {
    "normal": 4592,
    "transition": 1956
}
total_samples = sum(class_counts.values())
weights = torch.tensor(
    [total_samples / class_counts[action] for action in ["normal", "transition"]],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
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
