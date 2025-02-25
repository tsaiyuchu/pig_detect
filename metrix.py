import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_predictions(csv_file):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_file)

    # 檢查必要的欄位是否存在
    required_columns = ['True Label', 'Predicted Label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件必須包含以下欄位: {required_columns}")

    # 將 "normal" 和 "transition" 映射為數字
    label_map = {
        "normal": 0,
        "transition": 1
    }
    true_labels = df['True Label'].map(label_map)
    predicted_labels = df['Predicted Label'].map(label_map)

    # 檢查是否有未映射的標籤
    if true_labels.isnull().any() or predicted_labels.isnull().any():
        raise ValueError("存在無法映射的標籤，請檢查 True Label 或 Predicted Label 欄位是否有拼寫錯誤")

    # 計算 Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # 計算 Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # 計算 Classification Report
    class_report = classification_report(true_labels, predicted_labels, target_names=['normal', 'transition'])

    # 輸出結果
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy: {:.4f}".format(accuracy))
    print("\nClassification Report:")
    print(class_report)

    return conf_matrix, accuracy, class_report

# 使用範例
csv_file = "/home/yuchu/pig/metrix3.csv"  # 替換為你的 CSV 文件路徑
conf_matrix, accuracy, class_report = evaluate_predictions(csv_file)