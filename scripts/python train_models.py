# scripts/train_models.py
import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import pickle
import os

# Define data directories
base_data_dir = "C:/neuro_ai_project/data/"
mri_dir = os.path.join(base_data_dir, "mri")
genomics_dir = os.path.join(base_data_dir, "genomics")
clinical_dir = os.path.join(base_data_dir, "clinical")
models_dir = "C:/neuro_ai_project/models/"
os.makedirs(models_dir, exist_ok=True)

# Load MRI data
try:
    mri_images = np.load(os.path.join(mri_dir, "images.npy"))
    mri_labels = np.load(os.path.join(mri_dir, "labels.npy"))
    # Basic preprocessing (you might need more sophisticated methods)
    mri_images = mri_images / 255.0
    mri_labels = tf.keras.utils.to_categorical(mri_labels, num_classes=2)
    mri_images_train, mri_images_test, mri_labels_train, mri_labels_test = train_test_split(
        mri_images, mri_labels, test_size=0.2, random_state=42
    )
except FileNotFoundError:
    print("Error: MRI data files not found. Please run simulate_data.py first.")
    mri_images_train, mri_images_test, mri_labels_train, mri_labels_test = None, None, None, None

# Load clinical and genomics data
try:
    genomics_df = pd.read_csv(os.path.join(genomics_dir, "genomics_data.csv"))
    clinical_df = pd.read_csv(os.path.join(clinical_dir, "clinical_data.csv"))
    cg_df = pd.concat([clinical_df.drop('label', axis=1), genomics_df.drop('label', axis=1)], axis=1)
    cg_labels = genomics_df['label'] # Assuming labels are consistent
    cg_data_train, cg_data_test, cg_labels_train, cg_labels_test = train_test_split(
        cg_df, cg_labels, test_size=0.2, random_state=42
    )
except FileNotFoundError:
    print("Error: Clinical or genomics data files not found. Please run simulate_data.py first.")
    cg_data_train, cg_data_test, cg_labels_train, cg_labels_test = None, None, None, None

# Train MRI model (CNN)
if mri_images_train is not None:
    mri_model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    mri_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mri_model.fit(mri_images_train, mri_labels_train, epochs=5, validation_data=(mri_images_test, mri_labels_test), verbose=0)
    mri_loss, mri_accuracy = mri_model.evaluate(mri_images_test, mri_labels_test, verbose=0)
    print(f"MRI Model Test Accuracy: {mri_accuracy:.4f}")
    mri_model.save(os.path.join(models_dir, "mri_model.h5"))
else:
    print("Skipping MRI model training due to missing data.")
    mri_model = None

# Train clinical/genomic model (Random Forest)
if cg_data_train is not None:
    cg_model = RandomForestClassifier(random_state=42)
    cg_model.fit(cg_data_train, cg_labels_train)
    cg_predictions = cg_model.predict(cg_data_test)
    cg_accuracy = accuracy_score(cg_labels_test, cg_predictions)
    print(f"Clinical/Genomic Model Test Accuracy: {cg_accuracy:.4f}")
    with open(os.path.join(models_dir, "cg_model.pkl"), 'wb') as file:
        pickle.dump(cg_model, file)
else:
    print("Skipping Clinical/Genomic model training due to missing data.")
    cg_model = None

print("Model training completed and models saved.")