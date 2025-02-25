import numpy as np
from PIL import Image
import os

def npz_to_jpg(npz_file, output_folder):
    # 載入 npz 資料
    npz_data = np.load(npz_file)
    data = npz_data['data']    # shape: (N, H, W, C)
    filenames = npz_data['filenames']
    
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 將每張圖像存成 .jpg
    for img_array, fname in zip(data, filenames):
        # img_array: (H, W, C), uint8 格式資料
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        out_path = os.path.join(output_folder, fname)
        img.save(out_path, 'JPEG')
    print(f"已成功將 {npz_file} 中的影像存成 .jpg 至 {output_folder}")

# 範例用法
npz_to_jpg("/home/yuchu/pig/dataset/test.npz", "/home/yuchu/pig/dataset/test/")
#npz_to_jpg("/home/yuchu/pig/dataset/val.npz", "/home/yuchu/pig/dataset/restored_val_images")
