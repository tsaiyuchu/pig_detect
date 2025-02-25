import numpy as np
from PIL import Image
import os

def preprocess_and_save_to_npz(image_folder, output_file, image_size=(224, 224)):
    data = []
    filenames = []

    for img_name in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size)
        data.append(np.array(img))
        filenames.append(img_name)

    np.savez_compressed(output_file, data=np.array(data), filenames=filenames)

# Example usage:
preprocess_and_save_to_npz("/home/yuchu/pig/dataset/train", "/home/yuchu/pig/dataset/train.npz")
preprocess_and_save_to_npz("/home/yuchu/pig/dataset/val", "/home/yuchu/pig/dataset/val.npz")

