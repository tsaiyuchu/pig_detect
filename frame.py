import cv2
import os
import pandas as pd

# 設定參數
input_video = "/home/yuchu/pig/dataset/test5_stand.mp4"  # 原始影片的檔案路徑
output_base_folder = "/home/yuchu/pig/dataset/test5/"  # 幀圖片儲存的主資料夾
time_segments = [  # 定義多段時間範圍（單位：秒）
    {"start": 0, "end": 4}      # 第一段時間 14:32 - 16:02
    #{"start": 1258, "end": 1300},    # 第二段時間 20:58 - 21:40
    #{"start": 1442, "end": 1472},     # 第二段時間 24:02 - 24:32
    #{"start": 1537, "end": 1550},     # 第二段時間 25:37 - 25:50
    #{"start": 1830, "end": 2150},     # 第二段時間 30:30 - 35:50
    #{"start": 2433, "end": 2579}     # 第二段時間 40:33 - 42:59
]
fps = 30  # 每秒提取的幀數

# 創建主輸出資料夾
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# 打開影片
cap = cv2.VideoCapture(input_video)

# 獲取影片的資訊
video_fps = cap.get(cv2.CAP_PROP_FPS)  # 影片的幀率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 影片的總幀數
video_duration = frame_count / video_fps  # 影片的總時長（秒）

# 確保時間段有效
for segment in time_segments:
    if segment["start"] >= video_duration or segment["end"] > video_duration:
        print(f"指定的時段 {segment} 超出影片範圍")
        cap.release()
        exit()

# 開始處理每段時間
for i, segment in enumerate(time_segments):
    start_time = segment["start"]
    end_time = segment["end"]
    duration = end_time - start_time

    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    frame_interval = int(video_fps / fps)

    # 為當前段創建子資料夾
    segment_folder = os.path.join(output_base_folder, f"segment_{i+1}")
    os.makedirs(segment_folder, exist_ok=True)

    # 跳到開始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_frame_count = 0
    time_labels = []

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 只保存符合間隔的幀
        if (frame_idx - start_frame) % frame_interval == 0:
            frame_name = f"frame_{saved_frame_count:04d}.png"
            output_path = os.path.join(segment_folder, frame_name)
            cv2.imwrite(output_path, frame)
            time_labels.append(frame_name)
            saved_frame_count += 1

        frame_idx += 1

    # 創建 ODS 標籤檔案
    ods_file = os.path.join(segment_folder, f"segment_{i+1}.ods")
    df = pd.DataFrame({
        "time": time_labels,  # 幀圖片名稱
        "action": [""] * len(time_labels)  # 預設空的動作標籤
    })
    df.to_excel(ods_file, index=False, engine='odf')

    print(f"段 {i+1}: 提取完成，保存了 {saved_frame_count} 幀到資料夾 {segment_folder}")
    print(f"標籤檔案已生成：{ods_file}，請手動填寫 action 欄位")

# 釋放影片資源
cap.release()
print("所有段落處理完成！")
