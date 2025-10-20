# Applied Deep Learning Project

- **Student**: Balázs Róna
- **Project Type**: Bring Your Own Method

---

## 1. References

1. Zherzdev, S., & Gruzdev, A. (2018). *LPRNet: License Plate Recognition via Deep Neural Networks*. [arXiv:1806.10447](https://arxiv.org/abs/1806.10447)
2. Laroca, R., Severo, E., Zanlorensi, L. A., Oliveira, L. S., Gonçalves, G. R., Schwartz, W. R., & Menotti, D. (2018). *A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector*. [arXiv:1802.09567](https://arxiv.org/abs/1802.09567)

---

## 2. Topic

Automatic License Plate Segmentation and Recognition using Deep Neural Networks

---

## 3. Project Type: Bring Your Own Method

I will re-implement and modify a license plate detection and segmentation model (such as a U-Net or YOLOv8 segmentation head) and combine it with post-processing and OCR for end-to-end license plate reading.

---

## 4. Summary

### 4.1 Description

The goal is to build an end-to-end system for license plate recognition:

1. **Segment** license plates from real life images using a deep learning model.
2. **Post-process** the segmented region using image enhancement techniques (grayscale normalization, contrast stretching, thresholding).
3. **Extract text** from the enhanced region using a pre-trained OCR model (such as EasyOCR or Tesseract).

My approach is a modular deep-learning pipeline that focuses on accurate segmentation, since OCR performance is highly dependent on clean plate crops. Possible improvements include experimenting with segmentation models and better pre-/post-processing techniques.

---

### 4.2 Dataset

I plan to use the OpenALPR Benchmark Dataset or CCPD (Chinese City Parking Dataset), both of which are publicly available.

**Dataset characteristics:**

- ~200k images (CCPD)
- Bounding boxes or segmentation masks for plates
- Diverse illumination, angles, and backgrounds

---

### 4.3 Work-Breakdown Plan

| Task                                  | Description                                                                                              | Estimated Time |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------- |
| **1. Dataset Setup & Exploration**    | Download and inspect dataset, preprocess (resize, normalization, train/test split)                       | 5 h            |
| **2. Model Design & Implementation**  | Choose baseline (U-Net or YOLOv8-Seg), modify architecture (e.g., add attention or dropout improvements) | 10 h           |
| **3. Training & Fine-Tuning**         | Train on subset of dataset, tune hyperparameters, evaluate segmentation accuracy                         | 12 h           |
| **4. OCR Integration**                | Integrate pre-trained OCR model and connect to segmentation output                                       | 5 h            |
| **5. Post-Processing Pipeline**       | Image enhancement, thresholding, resizing for OCR                                                        | 3 h            |
| **6. Application / Demo Development** | Build small Python app to demonstrate full pipeline                                                      | 5 h            |
| **7. Final Report Writing**           | Document methods, results, and comparison with literature                                                | 6 h            |
| **8. Presentation Preparation**       | Prepare slides and demo for final presentation                                                           | 4 h            |

---

### 4.4 Expected Outcome

- A trained segmentation model capable of accurately detecting license plates.
- A working end-to-end pipeline combining segmentation, post-processing, and OCR.
- Evaluation of segmentation performance and OCR accuracy.
- A short demo of the system on test images.
