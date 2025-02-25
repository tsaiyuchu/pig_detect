import cv2
import pandas as pd
import os

def images_to_video_with_labels(image_folder, csv_file, output_video, sequence_length=5, fps=10):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file)
    if 'frame_file' not in df.columns or 'True Label' not in df.columns or 'Predicted Label' not in df.columns:
        raise ValueError("CSV 文件必須包含 'frame_file', 'True Label' 和 'Predicted Label' 欄位")

    # 確認圖片路徑
    images = df['frame_file'].tolist()
    true_labels = df['True Label'].tolist()
    predicted_labels = df['Predicted Label'].tolist()

    # 獲取第一張圖片的大小來設定影片解析度
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"無法讀取圖片：{first_image_path}")
    height, width, _ = first_image.shape

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 編碼格式
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX  # 字型
    font_scale = 1.2  # 字體大小
    font_thickness = 2  # 字體粗細
    text_color = (0, 255, 0)  # 綠色字體
    bg_color = (0, 0, 0)  # 黑色背景框

    # 依序將圖片寫入影片
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取圖片 {image_path}，將跳過該圖片")
            continue

        # 加入文字：True Label 和 Predicted Label
        true_label_text = f"True Label: {true_labels[i]}"
        pred_label_text = f"Prediction: {predicted_labels[i]}"

        # 繪製文字背景框
        (text_w1, text_h1), _ = cv2.getTextSize(true_label_text, font, font_scale, font_thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(pred_label_text, font, font_scale, font_thickness)
        cv2.rectangle(image, (5, 5), (max(text_w1, text_w2) + 20, text_h1 + text_h2 + 40), bg_color, -1)

        # 繪製文字
        cv2.putText(image, true_label_text, (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(image, pred_label_text, (10, 30 + text_h1 + 10), font, font_scale, text_color, font_thickness)

        # 將影像寫入影片
        video_writer.write(image)

    video_writer.release()
    print(f"影片已保存至 {output_video}")

# 範例執行
if __name__ == "__main__":
    image_folder = "/home/yuchu/pig/dataset/val1/"  # 測試圖片資料夾
    csv_file = "/home/yuchu/pig/test_results.csv"  # 包含 True Label 和 Predicted Label 的 CSV 檔案
    output_video = "test2.mp4"  # 輸出影片檔案名稱
    sequence_length = 5
    fps = 30  # 設定每秒幀數

    images_to_video_with_labels(image_folder, csv_file, output_video, sequence_length, fps)
