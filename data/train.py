import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import os
from PIL import Image
import numpy as np
import streamlit as st
import logging  # Import the logging module

# --- Constants ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # Adjust as needed
DATA_DIR = 'C:/neuro_ai_project/data'  # Root directory for your data
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_PATH = 'C:/neuro_ai_project/models/mri_model.h5'  # Where to save the trained model

# --- Set up logging ---
logging.basicConfig(level=logging.ERROR)  # Configure logging to capture errors

# --- Create Directories (if they don't exist) ---
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# --- Data Preparation ---
# This part assumes you have organized your data into 'train' and 'val' folders,
# each containing subfolders for each class (e.g., 'AD', 'CN').  If your data
# is organized differently, you'll need to adapt this section.
# Example directory structure:
# data/
# ├── train/
# │   ├── AD/
# │   │   ├── ad_image1.jpg
# │   │   ├── ad_image2.jpg
# │   │   └── ...
# │   ├── CN/
# │   │   ├── cn_image1.jpg
# │   │   ├── cn_image2.jpg
# │   │   └── ...
# ├── val/
# │   ├── AD/
# │   │   ├── ad_image3.jpg
# │   │   ├── ad_image4.jpg
# │   │   └── ...
# │   ├── CN/
# │   │   ├── cn_image3.jpg
# │   │   ├── cn_image4.jpg
# │   │   └── ...

def create_dummy_data(directory, num_images_per_class=10):
    """Creates dummy image data for demonstration purposes."""
    for class_name in ['AD', 'CN']:
        class_dir = os.path.join(directory, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images_per_class):
            img_path = os.path.join(class_dir, f'{class_name}_{i}.jpg')
            # Create a small black image
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            Image.fromarray(img).save(img_path)
    return True

if not os.listdir(TRAIN_DIR):
    st.warning(f"No training data found in {TRAIN_DIR}. Creating dummy data for demonstration.")
    create_dummy_data(TRAIN_DIR)
if not os.listdir(VAL_DIR):
    st.warning(f"No validation data found in {VAL_DIR}. Creating dummy data for demonstration.")
    create_dummy_data(VAL_DIR)
    

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=None,  # explicitly set this
    data_format='channels_last',  # explicitly set this
    validation_split=0.0 # explicitly set this
)
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # 'binary' for two classes, 'categorical' for more
        shuffle=True,  # Keep shuffle enabled
        seed=42,  # Add a seed for reproducibility
        #follow_links=True # Add this if you have symbolic links
    )
except Exception as e:
    logging.error(f"Error creating train_generator: {e}")
    train_generator = None  # Set to None to prevent further errors

val_datagen = ImageDataGenerator(
    rescale=1./255,  # Only rescale for validation
    preprocessing_function=None,
    data_format='channels_last'
)  
try:
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=42,
        #follow_links=True
    )
except Exception as e:
    logging.error(f"Error creating val_generator: {e}")
    val_generator = None

# --- Model Definition ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),  # Add dropout for regularization
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 'sigmoid' for binary, 'softmax' for multi-class
])

# --- Compile the Model ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 'binary_crossentropy' for two classes, 'categorical_crossentropy' for more
              metrics=['accuracy'])

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1)

# --- Train the Model ---
if train_generator and val_generator: # only train if generators are valid
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),  # Ensure at least 1 step
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=max(1, val_generator.samples // BATCH_SIZE),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
    except Exception as e:
        logging.error(f"Error during training: {e}")
        history = None
else:
    history = None
    logging.error("Training was skipped because of invalid data generators.")


# --- Save the Model (Optional -  ModelCheckpoint already saves the best model) ---
if history:
  model.save(MODEL_PATH)
  print(f'Trained model saved to {MODEL_PATH}')

# --- Plot Training History (Optional) ---
import matplotlib.pyplot as plt
if history is not None:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    print("Training history is None.  Model may not have trained.")
