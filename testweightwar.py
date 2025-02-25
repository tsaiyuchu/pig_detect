import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

# -------------- (1) 3D CNN 模型 (sequence_length=5) --------------
class PigAction3DCNN(nn.Module):
    def __init__(self, num_classes=2):  # 二分類
        super(PigAction3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        # 使用 Adaptive Pooling 將 D, H, W 固定 => D=1, H=56, W=56
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 56, 56))  
        
        self.fc1 = nn.Linear(64 * 1 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 3, sequence_length=5, 224, 224)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # x shape 可能是 (batch_size, 64, D', H', W')
        # Adaptive Pooling => (batch_size, 64, 1, 56, 56)
        x = self.adaptive_pool(x)

        # flatten
        x = x.view(x.size(0), -1)  # => (batch_size, 64*1*56*56) = (batch_size, 64*56*56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------- (2) 自定義 Dataset (讀取 CSV、.jpg, 序列長度=5) --------------
class PigDataset(Dataset):
    def __init__(self, image_folder, label_file, sequence_length=5, transform=None):
        self.image_folder = image_folder
        self.labels = pd.read_csv(label_file)
        self.labels.columns = self.labels.columns.str.strip()

        expected_columns = ['frame_file', 'action']
        if not all(col in self.labels.columns for col in expected_columns):
            raise ValueError(f"列名不匹配，當前列名為：{self.labels.columns}")

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
            img = self.transform(img)
            images.append(img)
        # shape: (sequence_length=5, C=3, H=224, W=224)
        images = torch.stack(images)
        # 以該序列「最後一張」的標籤作為序列標籤
        label = int(self.labels.iloc[idx + self.sequence_length - 1]['label'])
        return images, label

def get_test_loader(image_folder, label_file, batch_size=16, sequence_length=5):
    dataset = PigDataset(image_folder, label_file, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# -------------- (3) 測試並加上「預警區間」邏輯 --------------
def test_model(model_path, test_loader, device):
    model = PigAction3DCNN(num_classes=2)  # 二分類模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_samples = 0
    correct_predictions = 0

    all_predictions = []
    all_labels = []

    # 為了標記預警，需要知道測試集共有幾張 frame
    # => len(test_loader.dataset.labels) 就是 CSV 的總行數
    total_frames = len(test_loader.dataset.labels)
    # 每張圖片預設 warning="no"
    warning_list = ["no"] * total_frames

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # images shape: (batch_size, sequence_length=5, C=3, H=224, W=224)
            # 需轉成 (batch_size, 3, sequence_length, H, W)
            images = images.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 統計準確率
            total_samples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

            # 蒐集整體評估
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 依序處理每筆樣本，標記預警區間
            for i, p in enumerate(preds):
                # global_idx 代表 dataset 中的第幾筆序列
                global_idx = batch_idx * test_loader.batch_size + i
                # 該序列最後一張影像索引 => end_idx
                end_idx = global_idx + (test_loader.dataset.sequence_length - 1)
                # 若預測為 transition (== 1)，就把 [end_idx-15, end_idx] 標成 "yes"
                if p.item() == 1:
                    start_idx = end_idx - 15
                    if start_idx < 0:
                        start_idx = 0
                    # 將 [start_idx..end_idx] 設為 "yes"
                    for f in range(start_idx, end_idx+1):
                        if f < total_frames:
                            warning_list[f] = "yes"
                    # end_idx+1 設為 "no" (若不超出範圍)，
                    # 以便下一張恢復正常 (除非下一個序列又偵測到 transition)
                    if end_idx+1 < total_frames:
                        warning_list[end_idx+1] = "no"

    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.2f}%")
    return np.array(all_predictions), np.array(all_labels), warning_list, accuracy

# -------------- (4) 評估與儲存 --------------
def evaluate_results(predictions, labels, label_map):
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
    label_names = {v: k for k, v in label_map.items()}

    conf_matrix = confusion_matrix(labels, predictions, labels=list(label_map.values()))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    precision = precision_score(labels, predictions, labels=list(label_map.values()), average=None)
    recall = recall_score(labels, predictions, labels=list(label_map.values()), average=None)
    print("\nPrecision and Recall by Class:")
    for idx, label in enumerate(label_map.keys()):
        print(f"{label}: Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}")

    report = classification_report(labels, predictions, target_names=label_names.values(), digits=4)
    print("\nClassification Report:\n", report)

def save_results(predictions, labels, warnings, label_map, output_file="test_results.csv"):
    label_names = {v: k for k, v in label_map.items()}
    pred_names = [label_names.get(p, "unknown") for p in predictions]
    true_names = [label_names.get(l, "unknown") for l in labels]

    results = pd.DataFrame({
        "True Label": true_names,
        "Predicted Label": pred_names
        # 此處不包含 warnings，因為 warnings 是針對「每張 frame」的標記
        # 如果想給每個序列一個標記，也可放此 DataFrame
    })
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"測試結果已儲存至 {output_file}")

def save_warning_list(warning_list, output_file="warning_mark.csv"):
    df_warning = pd.DataFrame({
        "frame_index": list(range(len(warning_list))),
        "warning": warning_list
    })
    df_warning.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Warning 標記已輸出至 {output_file}")

# -------------- (5) 主程式 --------------
if __name__ == "__main__":
    # ----- 您可自行修改路徑參數 -----
    test_images_path = "/home/yuchu/pig/dataset/val"  
    test_labels_file = "/home/yuchu/pig/dataset/val.csv"
    model_path = "/home/yuchu/pig/weight/skbinary_model.pth"
    output_csv_results = "wtest_results.csv"
    output_csv_warning = "warning_mark.csv"
    
    batch_size = 64
    sequence_length = 5  # 維持舊架構的序列長度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立 DataLoader
    test_loader = get_test_loader(
        image_folder=test_images_path,
        label_file=test_labels_file,
        batch_size=batch_size,
        sequence_length=sequence_length
    )

    # 測試模型 & 取得預測與 warning
    label_map = { "normal": 0, "transition": 1 }
    predictions, labels, warning_list, accuracy = test_model(model_path, test_loader, device)

    # 評估
    evaluate_results(predictions, labels, label_map)

    # 儲存序列的 (True/Pred Label)
    save_results(predictions, labels, warning_list, label_map, output_file=output_csv_results)

    # 儲存每張影像的 warning 狀態
    save_warning_list(warning_list, output_file=output_csv_warning)
