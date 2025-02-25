import cv2
import os
import csv

def extract_frames_from_segments(video_path, segments, output_dir, csv_path):
    """
    從影片中依照給定的 (start_time, end_time) 秒數區間擷取影格，並儲存為圖片同時建立 CSV 檔案。

    Parameters:
    - video_path: 影片檔案路徑
    - segments: [(start_time, end_time), (start_time, end_time), ...] 單位為秒
    - output_dir: 輸出影像存放的目錄
    - csv_path: 最終輸出的標記用 CSV 檔路徑
    """

    # 建立輸出目錄（若不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 OpenCV 讀取影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 將提取出的影格與對應資訊寫入 CSV
    # CSV 欄位範例：video_id, start_time, end_time, frame_file, action(空白等之後標記)
    fields = ["video_id", "start_time", "end_time", "frame_file", "action"]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        # 依序處理每個指定的時間區段
        for seg_idx, (start_sec, end_sec) in enumerate(segments, start=1):
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            # 設定到開始擷取的 frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"在 {current_frame} 幀處無法讀取影像，可能已超出範圍或影片結束。")
                    break

                # 儲存影格圖片
                frame_filename = f"{video_name}_seg{seg_idx}_frame{current_frame}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                # 寫入 CSV 資訊，action 先留空白，之後手動標記
                writer.writerow([video_name, start_sec, end_sec, frame_filename, ""])

                current_frame += 1

    cap.release()
    print("擷取完成並建立 CSV:", csv_path)


if __name__ == "__main__":
    # 輸入參數範例
    video_path = "/home/yuchu/pig/dataset/train5_sut.mp4"  # 請修改為實際影片路徑
    segments = [
        (0, 20)

    ]
    output_dir = "/home/yuchu/pig/dataset/train4/"
    csv_path = "/home/yuchu/pig/dataset/train4.csv"

    extract_frames_from_segments(video_path, segments, output_dir, csv_path)
