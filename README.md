# Dental Radiography Image Classification  
## Comparative Analysis of Custom CNN, VGG19, and MobileNetV2

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Image%20Classification-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Dental%20Radiography-purple.svg)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
  - [Custom CNN](#1-custom-cnn)
  - [VGG19 Transfer Learning](#2-vgg19-transfer-learning)
  - [MobileNetV2 Transfer Learning](#3-mobilenetv2-transfer-learning)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)

---

## Project Overview

This project presents a comparative deep learning study for **dental radiographic image classification**.  
The goal is to classify cropped dental X-ray regions into one of four dental conditions:

- **Cavity**
- **Fillings**
- **Impacted Tooth**
- **Implant**

Three deep learning approaches were implemented and compared:

1. **Custom CNN** trained from scratch.
2. **VGG19** using transfer learning.
3. **MobileNetV2** using transfer learning.

The project focuses on the complete deep learning workflow, including annotation-based cropping, preprocessing, data augmentation, class imbalance handling, training, evaluation, and model comparison.

---

## Dataset

Dataset used:

```text
Dental Radiography Dataset
https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography
```

The original dataset contains dental radiography images with bounding-box annotations stored in:

```text
train/_annotations.csv
```

Each annotated object is cropped from the original image using its bounding-box coordinates:

```text
xmin, ymin, xmax, ymax
```

After cropping, images are organized into class-specific folders.

### Processed Dataset Summary

| Item | Count |
|---|---:|
| Total cropped samples | 8,028 |
| Training samples | 6,421 |
| Validation samples | 1,607 |
| Number of classes | 4 |

### Class Distribution

| Class | Total Samples |
|---|---:|
| Fillings | 5,239 |
| Implant | 1,785 |
| Cavity | 576 |
| Impacted Tooth | 428 |

The dataset is highly imbalanced, especially because the **Fillings** class has far more samples than the **Cavity** and **Impacted Tooth** classes.

---

## Problem Statement

Dental X-ray interpretation requires identifying subtle visual patterns such as cavities, fillings, implants, and impacted teeth.  
This project investigates whether a lightweight custom CNN can perform better than transfer-learning models pretrained on natural images.

The main questions are:

- Can a custom CNN learn dental-specific visual features effectively?
- How do VGG19 and MobileNetV2 perform when adapted to grayscale dental X-rays?
- How does class imbalance affect minority classes such as `Cavity` and `Impacted Tooth`?

---

## Preprocessing Pipeline

The preprocessing pipeline is one of the most important parts of this project.

### 1. Annotation-Based Cropping

The notebook reads bounding boxes from the annotation CSV file and crops the target dental region:

```python
crop = img.crop((xmin, ymin, xmax, ymax))
```

This makes the model focus on the annotated dental region instead of the entire X-ray.

### 2. Grayscale Conversion

All input images are converted to grayscale because dental radiographs are naturally grayscale.

### 3. CLAHE Enhancement

Contrast Limited Adaptive Histogram Equalization (**CLAHE**) is applied to improve local contrast:

```python
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

This helps make dental structures more visible.

### 4. Image Resizing

All images are resized to:

```text
224 x 224
```

### 5. Normalization

Pixel values are normalized from `[0, 255]` to `[-1, 1]`:

```python
image = image / 255.0
image = (image - 0.5) / 0.5
```

### 6. Data Augmentation

Training images are augmented using:

```python
RandomTranslation(0.1, 0.1)
RandomZoom(0.1)
RandomSharpness(0.5)
RandomContrast(0.8, 1.2)
```

Validation images are not augmented.

---

## Model Architectures

## 1. Custom CNN

The Custom CNN was built from scratch specifically for dental radiography images.

### Architecture

```text
Input: 224 x 224 x 1

Conv2D(32, 3x3, ReLU)
BatchNormalization
MaxPooling2D

Conv2D(64, 3x3, ReLU)
BatchNormalization
MaxPooling2D

Conv2D(128, 3x3, ReLU)
BatchNormalization
MaxPooling2D

Conv2D(256, 3x3, ReLU)
BatchNormalization

GlobalAveragePooling2D
Dense(128, ReLU)
Dropout(0.4)
Dense(4)
```

The final layer outputs **logits**, so softmax is applied only during prediction.

### Why Custom CNN?

The Custom CNN is smaller and more specialized than ImageNet-based models.  
Because dental X-rays are very different from natural RGB images, a custom model can learn dental-specific texture patterns more directly.

---

## 2. VGG19 Transfer Learning

VGG19 was used as a pretrained feature extractor with ImageNet weights.

### Main Setup

- Base model: `VGG19`
- Weights: `imagenet`
- Include top: `False`
- Base model frozen: `trainable = False`
- Input grayscale image repeated into 3 channels
- Classifier head:
  - `GlobalAveragePooling2D`
  - `Dense(256, relu)`
  - `Dropout(0.5)`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(4)`

### Why VGG19?

VGG19 is a strong transfer-learning baseline.  
However, it has many parameters and was pretrained on natural images, which may limit its performance on dental radiography images.

---

## 3. MobileNetV2 Transfer Learning

MobileNetV2 was used as a lightweight transfer-learning model.

### Main Setup

- Base model: `MobileNetV2`
- Weights: `imagenet`
- Include top: `False`
- Base model frozen: `trainable = False`
- Input grayscale image repeated into 3 channels
- Classifier head:
  - `GlobalAveragePooling2D`
  - `Dense(256, relu)`
  - `Dropout(0.5)`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(4)`

### Why MobileNetV2?

MobileNetV2 is efficient and suitable for lightweight deployment.  
It uses depthwise separable convolutions, making it faster and smaller than VGG19.

---

## Training Strategy

All models were trained using the same general training setup.

| Setting | Value |
|---|---|
| Image size | 224 x 224 |
| Batch size | 32 |
| Epochs | Up to 40 |
| Loss | Sparse Categorical Crossentropy |
| Output | Logits |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 5 |
| LR scheduler factor | 0.3 |
| Minimum LR | 1e-6 |

### Class Weights

Class weighting was used to reduce the effect of class imbalance.

| Class | Weight |
|---|---:|
| Cavity | 3.4897 |
| Fillings | 0.3830 |
| Impacted Tooth | 4.6937 |
| Implant | 1.1241 |

The formula used:

```text
class_weight = total_samples / (number_of_classes * class_count)
```

---

## Results

### Overall Comparison

| Model | Validation Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Custom CNN | **95.08%** | **0.9174** | **0.9502** |
| VGG19 | 87.06% | 0.7772 | 0.8651 |
| MobileNetV2 | 86.87% | 0.7746 | 0.8712 |

The **Custom CNN achieved the best overall validation performance**.

---

## Per-Class F1-Score Comparison

| Class | Custom CNN | VGG19 | MobileNetV2 |
|---|---:|---:|---:|
| Cavity | **0.8815** | 0.6354 | 0.5971 |
| Fillings | **0.9678** | 0.9112 | 0.9117 |
| Impacted Tooth | **0.8834** | 0.7237 | 0.7092 |
| Implant | **0.9370** | 0.8385 | 0.8805 |

---

## Detailed Classification Reports

### Custom CNN

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cavity | 0.9789 | 0.8017 | 0.8815 | 116 |
| Fillings | 0.9614 | 0.9742 | 0.9678 | 1048 |
| Impacted Tooth | 0.9351 | 0.8372 | 0.8834 | 86 |
| Implant | 0.9169 | 0.9580 | 0.9370 | 357 |

### VGG19

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cavity | 0.8026 | 0.5259 | 0.6354 | 116 |
| Fillings | 0.8718 | 0.9542 | 0.9112 | 1048 |
| Impacted Tooth | 0.8333 | 0.6395 | 0.7237 | 86 |
| Implant | 0.8899 | 0.7927 | 0.8385 | 357 |

### MobileNetV2

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cavity | 0.5123 | 0.7155 | 0.5971 | 116 |
| Fillings | 0.9170 | 0.9065 | 0.9117 | 1048 |
| Impacted Tooth | 0.9091 | 0.5814 | 0.7092 | 86 |
| Implant | 0.8842 | 0.8768 | 0.8805 | 357 |

---

## Key Findings

- The **Custom CNN** achieved the highest validation accuracy at **95.08%**.
- The Custom CNN performed best across all four classes.
- VGG19 performed better than MobileNetV2 on `Cavity` F1-score.
- MobileNetV2 achieved a slightly better weighted F1-score than VGG19.
- Transfer learning models were strong baselines, but their ImageNet features were not as effective as a dental-specific CNN.
- CLAHE preprocessing helped enhance local contrast in dental radiography images.
- Class weighting helped handle the strong imbalance between `Fillings` and the smaller classes.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/dental-radiography-classification.git
cd dental-radiography-classification
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography
```

Place the dataset in the expected project directory.

### 4. Run the Notebooks

Recommended order:

```text
1. CNN_Project_Convert.ipynb
2. CNN-Project Convert.ipynb
3. Mobilenetv2_Project.ipynb
```

Or rename them clearly:

```text
custom_cnn_dental_radiography.ipynb
vgg19_dental_radiography.ipynb
mobilenetv2_dental_radiography.ipynb
```

---

## Repository Structure

Recommended final GitHub structure:

```text
dental-radiography-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── custom_cnn_dental_radiography.ipynb
│   ├── vgg19_dental_radiography.ipynb
│   └── mobilenetv2_dental_radiography.ipynb
│
├── results/
│   ├── class_distribution.png
│   ├── sample_crops.png
│   ├── custom_cnn_accuracy_loss.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── vgg19_accuracy_loss.png
│   ├── vgg19_confusion_matrix.png
│   ├── mobilenetv2_accuracy_loss.png
│   └── mobilenetv2_confusion_matrix.png
│
├── models/
│   ├── best_model.weights.h5
│   ├── cnn_model.weights.h5
│   └── cnn_history.json
│
└── docs/
    └── AI_Strategies_Report.pdf
```

---

## Recommended Files to Upload to GitHub

Upload:

```text
README.md
requirements.txt
notebooks/
results/
docs/
```

For model weights, you have two options:

### Option 1: Upload weights directly

This is fine if the files are small.

### Option 2: Use Git LFS

Recommended for `.h5` files:

```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add .
git commit -m "Add dental radiography classification project"
git push
```

---

## Requirements

Example `requirements.txt`:

```text
tensorflow
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```

---

## Limitations

- The model was evaluated on a validation split from the same dataset, not on an external clinical dataset.
- The project uses cropped annotated regions, so full X-ray images require a detection/cropping step before classification.
- The dataset is imbalanced, especially toward the `Fillings` class.
- The models are for educational and research purposes only, not clinical diagnosis.

---

## Future Work

- Add a dedicated external test set.
- Fine-tune VGG19 and MobileNetV2 instead of keeping the base models fully frozen.
- Try medical-image pretrained backbones.
- Add Grad-CAM visualizations to explain model decisions.
- Build a small web demo using Streamlit or Gradio.
- Save full `.keras` models for easier reuse.
- Improve minority-class performance using stronger augmentation or focal loss.

---

## Team

- Shady Wafik Heshmat
- Youssef Mohamed Eladl
- Mostafa Farag

---

## Acknowledgment

Special thanks to:

- Dr. Mohammed Marey
- TA. Ahmed Tarek
