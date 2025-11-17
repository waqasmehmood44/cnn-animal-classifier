# Importing required libraries
import os                          # For interacting with the operating system's file structure
import shutil                      # For copying files and directories
import random                      # For shuffling lists (used in dataset splitting)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading and augmenting image data
from tensorflow.keras.applications import MobileNetV2                # Pre-trained MobileNetV2 model
from tensorflow.keras.models import Sequential                       # Sequential model API from Keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Layers used for custom classification head
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint        # Callbacks for training process
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Define paths for dataset and model saving
BASE_DATA_DIR = r'H:\ML Final Exam Project\animals\animals'                   # Original dataset path (contains folders for each animal class)
SPLIT_BASE_DIR = r'H:\ML Final Exam Project\animals\split_dataset'            # Path to save split dataset (train/val)
TRAIN_DIR = os.path.join(SPLIT_BASE_DIR, 'train')                  # Training dataset directory
VAL_DIR = os.path.join(SPLIT_BASE_DIR, 'val')                      # Validation dataset directory
MODEL_PATH = r'H:\ML Final Exam Project\model\cnn_animal_classifier.h5'       # Path to save trained model
ANIMAL_NAMES_PATH = r'H:\ML Final Exam Project\animals\animalnames.txt'       # Path to save class label names

# Function to split dataset into training and validation sets
def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Check if the dataset has already been split
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Train/val folders already exist. Skipping dataset split.")
        return

    print("Splitting dataset into train and validation...")
    
    # Loop through each category/class folder in source_dir
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip if not a directory

        # List and shuffle all image files in this category
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        random.shuffle(images)
        
        # Split index to divide into train and val sets
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create destination folders
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
        # Copy validation images
        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))
    
    print("Dataset split completed.")

# Perform the split
split_dataset(BASE_DATA_DIR, TRAIN_DIR, VAL_DIR)

# Define image size and training parameters
IMG_SIZE = (224, 224)        # Image input size expected by MobileNetV2
BATCH_SIZE = 16              # Number of images per batch during training
EPOCHS = 20                  # Max number of training epochs

# Data augmentation for training set to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values to [0, 1]
    rotation_range=20,           # Randomly rotate images by 20 degrees
    width_shift_range=0.1,       # Horizontal translation
    height_shift_range=0.1,      # Vertical translation
    zoom_range=0.1,              # Random zoom
    horizontal_flip=True,        # Random horizontal flip
    fill_mode='nearest'          # Fill in pixels after transformations
)

# Validation data should not be augmented, just normalized
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and iterate training images from directory
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'      # Use categorical labels (one-hot vectors)
)

# Load and iterate validation images from directory
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save label map to a text file for later use (e.g., prediction mapping)
with open(ANIMAL_NAMES_PATH, 'w') as f:
    label_map = {v: k for k, v in train_generator.class_indices.items()}  # Reverse mapping from class index to class name
    for i in range(len(label_map)):
        f.write(f"{label_map[i]}\n")  # Write each label on a new line

# Load MobileNetV2 without top classification layers
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained weights during initial training

# Build the classification head on top of base model
model = Sequential([
    base_model,                       # Use MobileNetV2 as a feature extractor
    GlobalAveragePooling2D(),        # Reduce feature maps to a single vector per image
    Dense(128, activation='relu'),   # Dense layer for learning non-linear combinations
    Dropout(0.5),                    # Dropout for regularization (reduce overfitting)
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with one unit per class
])

# Compile the model with optimizer, loss, and evaluation metric
model.compile(
    optimizer='adam',                           # Adam optimizer for adaptive learning
    loss='categorical_crossentropy',            # Loss function for multi-class classification
    metrics=['accuracy']                        # Evaluate performance using accuracy
)

# Define callbacks to enhance training process
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop early if no val_loss improvement
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)  # Save model only if val_accuracy improves

# Train the model using training and validation data
print("Training model...")
model.fit(
    train_generator,
    epochs=EPOCHS,                              
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint],
    verbose=1   # 👈 Enables live progress bar per batch
)


# Final message after training completes
print(f"Training complete. Best model saved at {MODEL_PATH}")
