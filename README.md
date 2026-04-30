# Dental Radiography Classification

A comparative deep learning project for classifying dental radiography images into four categories:

- Cavity
- Fillings
- Impacted Tooth
- Implant

The project compares three TensorFlow/Keras models:

1. Custom CNN
2. VGG19 Transfer Learning
3. MobileNetV2 Transfer Learning

Dataset:  
https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography

---

## Repository Structure

```text
Dental-Radiography-Classification/
│
├── README.md
│
├── archive.zip (Dataset imported from kaggle)
│
│
├── Custom_CNN/
│   ├── CNN_Project_Convert.ipynb
│   ├── best_model.weights.h5
│   ├── cnn_model.weights.h5
│   └── cnn_history.json
│
├── VGG19/
│   ├── VGG19_Project.ipynb
│   ├── best_model.weights.h5
│   ├── cnn_model.weights.h5
│   └── cnn_history.json
│
└── MobileNetV2/
    ├── Mobilenetv2_Project.ipynb
    ├── best_model.weights.h5
    ├── cnn_model.weights.h5
    └── cnn_history.json
```

Each model folder contains:

| File | Description |
|---|---|
| `*.ipynb` | Notebook containing preprocessing, training, and evaluation |
| `best_model.weights.h5` | Best saved model weights |
| `cnn_model.weights.h5` | Final saved model weights |
| `cnn_history.json` | Training history |

---

## Dataset Preparation

The dataset contains dental X-ray images with annotations.  
Each image region is cropped using bounding box coordinates, then assigned to one of the four classes.

The processed images are split into:

| Split | Samples |
|---|---:|
| Training | 6,421 |
| Validation | 1,607 |

Total processed samples: **8,028 images**

---

## Preprocessing

The preprocessing pipeline includes:

- Cropping images using bounding box annotations
- Converting images to grayscale
- Applying CLAHE for contrast enhancement
- Resizing images to `224 x 224`
- Normalizing pixel values to the range `[-1, 1]`
- Applying data augmentation during training

---

## Models

### 1. Custom CNN

A convolutional neural network trained from scratch.

Main layers:

```text
Conv2D -> BatchNormalization -> MaxPooling
Conv2D -> BatchNormalization -> MaxPooling
Conv2D -> BatchNormalization -> MaxPooling
Conv2D -> BatchNormalization
GlobalAveragePooling
Dense
Dropout
Output Layer
```

### 2. VGG19

A transfer learning model using a pretrained VGG19 backbone.

The base model is frozen and a custom classification head is added.

### 3. MobileNetV2

A lightweight transfer learning model using a pretrained MobileNetV2 backbone.

The base model is frozen and a custom classification head is added.

---

## Results

| Model | Accuracy | Cavity F1 | Fillings F1 | Impacted Tooth F1 | Implant F1 |
|---|---:|---:|---:|---:|---:|
| Custom CNN | **95.08%** | **0.8815** | **0.9678** | **0.8834** | **0.9370** |
| VGG19 | 87.06% | 0.6354 | 0.9112 | 0.7237 | 0.8385 |
| MobileNetV2 | 86.87% | 0.5971 | 0.9117 | 0.7092 | 0.8805 |

The Custom CNN achieved the best overall performance on the validation set.

---

## How to Use

Open any notebook and run it in Google Colab or Jupyter Notebook.

Recommended order:

1. `Custom_CNN/CNN_Project_Convert.ipynb`
2. `VGG19/VGG19_Project.ipynb`
3. `MobileNetV2/Mobilenetv2_Project.ipynb`

Install the main dependencies:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn pillow
```

---

## Notes

- The `.h5` files are weights files, not complete standalone models.
- To reuse a model, the same architecture must be rebuilt first, then the weights can be loaded.
- `best_model.weights.h5` is the recommended file for evaluation because it stores the best checkpoint.
- The project is intended for learning and experimentation, not clinical diagnosis.

---

## Future Improvements

- Add external test data
- Save complete `.keras` models
- Add Grad-CAM visualization
- Fine-tune VGG19 and MobileNetV2
- Build a simple web demo
