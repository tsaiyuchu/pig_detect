import os
import ezodf

def read_ods_ezodf(file_path):
    try:
        doc = ezodf.opendoc(file_path)
        sheet = doc.sheets[0]
        data = []
        for row in sheet.rows():
            row_data = [cell.value for cell in row]
            if any(row_data):  # 只添加非空行
                data.append(row_data)
        return data
    except Exception as e:
        print(f"Error reading {file_path} with ezodf: {e}")
        return None

def verify_images(folder_paths):
    for folder in folder_paths:
        ods_file = os.path.join(folder, "labels.ods")
        images_folder = os.path.join(folder, "frame")
        print(f"Verifying images for: {ods_file}")
        
        # 讀取 ODS 文件
        data = read_ods_ezodf(ods_file)
        if not data:
            print(f"Failed to read {ods_file}. Skipping.")
            continue

        headers, rows = data[0], data[1:]
        if "time" not in headers:
            print(f"'time' column not found in {ods_file}. Skipping.")
            continue

        time_index = headers.index("time")
        missing_images = []

        for row in rows:
            image_file = os.path.join(images_folder, row[time_index])
            if not os.path.exists(image_file):
                missing_images.append(row[time_index])

        if missing_images:
            print(f"Missing images in {images_folder}: {missing_images}")
        else:
            print(f"All images are present in {images_folder}.")

# 測試每個 labels.ods 文件
folders = [
    "/home/yuchu/pig/dataset/segment_1/",
    "/home/yuchu/pig/dataset/segment_2/",
    "/home/yuchu/pig/dataset/segment_3/",
    "/home/yuchu/pig/dataset/segment_4/",
    "/home/yuchu/pig/dataset/segment_5/",
    "/home/yuchu/pig/dataset/segment_6/",
]

for folder in folders:
    ods_file = os.path.join(folder, "labels.ods")
    print(f"Checking file: {ods_file}")
    if not os.path.exists(ods_file):
        print(f"File not found: {ods_file}")
        continue

    data = read_ods_ezodf(ods_file)
    if data:
        print(f"File {ods_file} loaded successfully. First 5 rows:")
        for row in data[:5]:
            print(row)
    else:
        print(f"Failed to read file: {ods_file}")

verify_images(folders)