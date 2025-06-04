# 🧠 AI & Image Processing Lab Series – University Coursework

This repository contains 5 detailed lab reports and implementations, completed as part of the **Digital Image & Video Processing** course at **University of Science – VNUHCM (HCMUS)**. Each lab focuses on a key aspect of image processing, neural networks, or deep learning frameworks like PyTorch and YOLO.

---

## 📁 Lab Overview

| Lab | Topic                              | Description |
|-----|------------------------------------|-------------|
| 1   | Image Processing with OpenCV       | Linear/Non-linear color transforms, histogram equalization, affine & bilinear transforms, filtering and smoothing using custom OpenCV code. |
| 2   | Edge Detection                     | Custom implementations of Sobel, Prewitt, Robert, Frei-Chen, Laplace, Laplacian of Gaussian, and full 7-step Canny edge detection (including NMS, hysteresis). |
| 3   | PyTorch Practice #1                | Implemented a simple FFNN (feed-forward neural network) from scratch using PyTorch, including forward/backward pass, training, predicting, saving/loading weights. |
| 4   | PyTorch Practice #2                | Modular neural network from scratch using custom layers (`FCLayer`, `ActivationLayer`, `Network`) with XOR dataset, tanh/sigmoid activations, and MSE loss. |
| 5   | YOLO Object Detection (YOLOv7 + YOLOv11) | Training YOLOv7 and YOLOv11 on a custom 12-class chess piece dataset using Roboflow and Ultralytics. Covers dataset preparation, model config, training, evaluation, and result analysis. |

---

## 🔍 Key Highlights

### 🧪 Image Processing
- Grayscale image reading
- Linear, logarithmic, exponential pixel mappings
- Histogram Equalization
- Affine & Bilinear geometric transforms
- Nearest-neighbor & linear interpolation
- Averaging, Gaussian, Median filters and Gaussian blur

### 📐 Edge Detection
- Manual implementation of gradient-based and Laplacian-based methods
- Full Canny pipeline: Gaussian blur → gradient → NMS → thresholding → edge linking
- Performance comparison with OpenCV versions (accuracy, speed, maintainability)

### 🔬 Neural Networks (PyTorch Labs)
- Feedforward neural net from scratch
- Modular layer architecture (network, loss, activation)
- XOR classification using tanh/sigmoid
- Manual forward/backward pass logic
- Experiments with learning rate, epoch count, and number of classes

### 📦 Object Detection (YOLO)
- Used Roboflow dataset of chess pieces (640x640, 12 classes)
- Trained YOLOv7 with custom `.yaml` config
- Used YOLOv11 (Ultralytics CLI) for efficient object detection
- Compared model performance (YOLOv11 > YOLOv7)
- Visualizations, precision, recall, and mAP metrics included

---

## ▶️ How to Run

Each lab has its own `source/` folder. You can:

```bash
cd "lab 1/source"
python 21127690_lab01.py
```

For PyTorch labs:

```bash
cd "lab 3/source"
python test_ffnn.py
```

For YOLO labs:

- Follow detailed steps in reports (`lab 5/docs/`)
- Training was done in Google Colab with dataset from Roboflow
- YOLOv11 CLI examples included

---

## 📑 Reports

All lab reports are included in `.docx` and `.pdf` formats per lab (`doc/`, `docs/`, or `report/` folders), covering:

- Function list
- Usage summary
- Pseudo code
- Experiment results
- Observations & evaluations

---

## 📌 Dependencies

Install commonly used packages:

```bash
pip install numpy matplotlib opencv-python torch torchvision ultralytics
```

*YOLO labs were run in Google Colab with CUDA-enabled GPUs.*

---

## 👨‍🎓 Author

GitHub: [wa1mpls](https://github.com/wa1mpls)

---

## 🏁 Note

All labs were implemented from scratch with limited use of high-level libraries (especially in Labs 1–4). This repository serves as a comprehensive showcase of **image processing, neural network engineering**, and **deep learning practice with real data**.
