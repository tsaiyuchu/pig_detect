# pig_detect

## **1. Overview**
This project aims to detect and classify pig actions, including **transmition (movement), standing, and sitting**, using a **3D CNN-based model**. The pipeline involves extracting pig frames from videos, segmenting specific action sequences, and training a deep learning model to recognize these actions.

---

## **2. Workflow**
### **Step 1: Video Processing & Action Segmentation**
1. **Extract Pig Frames**:
   - Use OpenCV or FFmpeg to process raw videos.
   - Crop or extract only the pig regions to remove unnecessary background.
   
2. **Segment Specific Actions**:
   - Identify frames corresponding to **transmition (movement), standing, and sitting**.
   - Clip and save sequences into separate labeled datasets.
   - Store data in structured folders:  
     ```
     dataset/
     ├── transmition/
     │   ├── video_001.mp4
     │   ├── video_002.mp4
     ├── standing/
     │   ├── video_003.mp4
     │   ├── video_004.mp4
     ├── sitting/
     │   ├── video_005.mp4
     │   ├── video_006.mp4
     ```
   - Optional: Use Optical Flow to enhance motion-based feature extraction.

### **Step 2: Preprocessing**
1. **Convert Videos into Frames**:
   - Convert each video into sequences of images.
   - Resize frames to a fixed resolution (e.g., `128x128`).
   - Normalize pixel values between `[0,1]` or `[-1,1]`.
   - Store as `.npy` or `.h5` format for efficient processing.

2. **Generate 3D Input Data**:
   - Stack frames as tensors with shape `[Frames, Height, Width, Channels]`.
   - Define a fixed frame length per sample (e.g., 16 or 32 frames per sequence).
   
---

### **Step 3: 3D CNN Training**
1. **Model Architecture**:
   - Use a **3D Convolutional Neural Network (3D CNN)** such as **C3D, I3D, R(2+1)D, or SlowFast**.
   - Input: Video sequences as tensors `[Batch, Frames, Height, Width, Channels]`.
   - Output: Multi-class classification (`transmition`, `standing`, `sitting`).

2. **Training Strategy**:
   - Split dataset into **Training (80%), Validation (10%), and Test (10%)**.
   - Use **data augmentation** (e.g., horizontal flipping, brightness adjustments, temporal jittering).
   - Apply **learning rate scheduling** and **early stopping** for better convergence.
   - Train using **Cross-Entropy Loss + Adam Optimizer**.

3. **Evaluation**:
   - Compute accuracy, precision, recall, and F1-score.
   - Generate confusion matrix to analyze misclassifications.
   - Test model on unseen video clips.

---
![image](https://github.com/user-attachments/assets/d5d75a53-b859-4fc5-a46a-ad812c7d85b4)
![image](https://github.com/user-attachments/assets/af77a5c8-d5f5-4c5a-99bb-7553ef0fd705)
