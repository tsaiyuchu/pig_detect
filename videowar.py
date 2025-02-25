import cv2
import pandas as pd
import os

def images_to_video_with_labels(image_folder, csv_file, output_video, sequence_length=5, fps=10):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file)
    required_columns = ['frame_file', 'True Label', 'Predicted Label', 'Warning']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件必須包含以下欄位: {required_columns}")
    
    # 確認圖片路徑
    images = df['frame_file'].tolist()
    true_labels = df['True Label'].tolist()
    predicted_labels = df['Predicted Label'].tolist()
    warnings = df['Warning'].astype(str).tolist()  # 將 Warning 欄位轉換為字串
    
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
    text_color_normal = (0, 255, 0)  # 綠色字體
    text_color_warning = (0, 0, 255)  # 紅色字體
    bg_color = (0, 0, 0)  # 黑色背景框
    
    # 依序將圖片寫入影片
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取圖片 {image_path}，將跳過該圖片")
            continue
    
        # 加入文字：True Label、Predicted Label 和 Warning
        true_label_text = f"True Label: {true_labels[i]}"
        pred_label_text = f"Prediction: {predicted_labels[i]}"
        warning_text = f"Warning: {warnings[i]}"  # 新增 Warning 文字
    
        # 繪製文字背景框
        (text_w1, text_h1), _ = cv2.getTextSize(true_label_text, font, font_scale, font_thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(pred_label_text, font, font_scale, font_thickness)
        (text_w3, text_h3), _ = cv2.getTextSize(warning_text, font, font_scale, font_thickness)  # 計算 Warning 文字大小
        
        max_text_width = max(text_w1, text_w2, text_w3)
        total_text_height = text_h1 + text_h2 + text_h3 + 60  # 增加額外的高度以容納 Warning
        
        cv2.rectangle(image, 
                      (5, 5), 
                      (max_text_width + 20, total_text_height), 
                      bg_color, 
                      -1)
    
        # 繪製文字
        cv2.putText(image, true_label_text, (10, 30), font, font_scale, text_color_normal, font_thickness)
        cv2.putText(image, pred_label_text, (10, 30 + text_h1 + 10), font, font_scale, text_color_normal, font_thickness)
        
        # 根據 Warning 狀態設定文字顏色
        warning_value = warnings[i].strip().lower()
        if warning_value in ['yes', 'true', '1']:  # 根據您的 Warning 格式調整條件
            warning_color = text_color_warning
        else:
            warning_color = text_color_normal
        
        cv2.putText(image, warning_text, (10, 30 + text_h1 + text_h2 + 20), font, font_scale, warning_color, font_thickness)
    
        # 將影像寫入影片
        video_writer.write(image)
    
    video_writer.release()
    print(f"影片已保存至 {output_video}")

# 範例執行
if __name__ == "__main__":
    image_folder = "/home/yuchu/pig/dataset/val"  # 測試圖片資料夾
    csv_file = "/home/yuchu/pig/wtest_results.csv"  # 包含 True Label、Predicted Label 和 Warning 的 CSV 檔案
    output_video = "testwarning.mp4"  # 輸出影片檔案名稱
    sequence_length = 5
    fps = 30  # 設定每秒幀數
    
    images_to_video_with_labels(image_folder, csv_file, output_video, sequence_length, fps)
