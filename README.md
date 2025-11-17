# 🐾 Animal Image Classifier (CNN)

A deep‑learning project using a Convolutional Neural Network (CNN) to classify animal images. This repository contains the trained TensorFlow/Keras model, preprocessing utilities, and a Python script for generating predictions on test images with confidence scores and visualization.

---

## 📌 Features

* ✔️ Trained CNN model (`.h5`)
* ✔️ Support for multiple animal classes
* ✔️ Script for loading images and generating predictions
* ✔️ Image visualization with predicted labels & confidence
* ✔️ Easy‑to‑modify code structure

---

## 📂 Project Structure

```
project-folder/
│
├── model/
│   └── cnn_animal_classifier.h5
│
├── animals/
│   ├── animalnames.txt
│   └── test_images/
│        ├── image1.jpg
│        ├── image2.jpg
│        └── ...
│
├── predict.py
└── README.md
```

---

## 🧠 Model Information

* Framework: **TensorFlow / Keras**
* Input Size: **224 × 224 × 3**
* Output: **Softmax probabilities for all animal classes**
* Labels loaded from: `animalnames.txt`

---

## ▶️ How to Run Predictions

### **1. Install Dependencies**

```bash
pip install tensorflow matplotlib numpy
```

### **2. Modify Paths (if needed)**

In your script:

```python
MODEL_PATH = r"path/to/cnn_animal_classifier.h5"
TEST_DIR = r"path/to/test_images"
ANIMAL_NAMES_PATH = r"path/to/animalnames.txt"
```

### **3. Run the Script**

```bash
python predict.py
```

The script will:

* Load the CNN model
* Read label names
* Predict each test image
* Display output image with prediction and confidence

---

## 📸 Example Output

**Console:**

```
prediction for tiger.jpg:
  Tiger: 0.9823
```

**Visualization:**

* Image pops up with title:
  **Prediction: Tiger (98.23%)**

---

## 📑 Script Overview

Your main prediction script:

* Loads the model using `load_model()`
* Resizes image to 224x224
* Converts to array and normalizes
* Predicts using softmax
* Displays image with matplotlib

---

## 📝 Label File Format

`animalnames.txt` must contain **one label per line**:

```
Cat
Dog
Elephant
Tiger
Lion
```

---

## 🧪 Test Images

Place all images you want to evaluate into:

```
animals/test_images/
```

File types supported:

* `.jpg`
* `.jpeg`
* `.png`

---

## 🔧 Customization

You can easily modify:

* Image size
* Model path
* Output format
* Top‑k predictions
* Batch prediction

If you need help adding these features, just ask!

---

## 📜 License

This project is for educational and experimental use.

---

## 💬 Support

If you'd like:

* a training script
* dataset preparation guide
* requirements file
* improvements to the README

Just let me know!
