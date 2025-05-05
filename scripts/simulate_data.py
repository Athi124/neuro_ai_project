# scripts/simulate_data.py
import numpy as np # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore
import os

# Define the base data directory
base_data_dir = "C:/neuro_ai_project/data/"
mri_dir = os.path.join(base_data_dir, "mri")
genomics_dir = os.path.join(base_data_dir, "genomics")
clinical_dir = os.path.join(base_data_dir, "clinical")

# Create directories if they don't exist
os.makedirs(mri_dir, exist_ok=True)
os.makedirs(genomics_dir, exist_ok=True)
os.makedirs(clinical_dir, exist_ok=True)

# Simulate MRI images (very basic)
def simulate_mri(num_images=100, size=64, num_classes=2):
    images = np.random.rand(num_images, size, size, 1).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_images)
    return images, labels

# Simulate genomics data
def simulate_genomics(num_samples=100, num_features=50, num_classes=2):
    data = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, num_classes, num_samples)
    return pd.DataFrame(data), pd.Series(labels)

# Simulate clinical data
def simulate_clinical(num_samples=100, num_features=10, num_classes=2):
    data = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, num_classes, num_samples)
    return pd.DataFrame(data), pd.Series(labels)

if __name__ == "__main__":
    num_samples = 100
    mri_images, mri_labels = simulate_mri(num_images=num_samples)
    genomics_data, genomics_labels = simulate_genomics(num_samples=num_samples)
    clinical_data, clinical_labels = simulate_clinical(num_samples=num_samples)

    # Save simulated MRI (as numpy arrays)
    np.save(os.path.join(mri_dir, "images.npy"), mri_images)
    np.save(os.path.join(mri_dir, "labels.npy"), mri_labels)

    # Save simulated genomics and clinical data as CSV
    genomics_data['label'] = genomics_labels
    genomics_data.to_csv(os.path.join(genomics_dir, "genomics_data.csv"), index=False)
    clinical_data['label'] = clinical_labels
    clinical_data.to_csv(os.path.join(clinical_dir, "clinical_data.csv"), index=False)

    print("Simulated data generated and saved.")