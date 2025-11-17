import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
MODEL_PATH = r'H:\ML Final Exam Project\model\cnn_animal_classifier.h5'
TEST_DIR = r'H:\ML Final Exam Project\animals\test_images'
ANIMAL_NAMES_PATH = r'H:\ML Final Exam Project\animals\animalnames.txt'

# Image dimensionsnow
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load label names
readable_labels = []
if os.path.exists(ANIMAL_NAMES_PATH):
    with open(ANIMAL_NAMES_PATH, 'r') as f:
        readable_labels = [line.strip() for line in f]

# Load the trained model
print("Loading model...")
model = load_model(MODEL_PATH)

# Prediction function for a single image (Top-1 only)
def predict_image(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top1_idx = np.argmax(prediction)

    label = readable_labels[top1_idx] if top1_idx < len(readable_labels) else f"Class {top1_idx}"
    confidence = prediction[top1_idx]

    print(f"\nprediction for {os.path.basename(img_path)}:")
    print(f"  {label}: {confidence:.4f}")

    # Display image with prediction as title
    plt.imshow(load_img(img_path))
    plt.title(f"Prediction: {label} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

# Run predictions on all test images
if os.path.exists(TEST_DIR):
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_files:
        print(f"No image files found in {TEST_DIR}.")
    else:
        for file in test_files:
            print(f"Predicting {file} ...")
            predict_image(os.path.join(TEST_DIR, file))
else:
    print(f"Test directory not found at {TEST_DIR}.")
