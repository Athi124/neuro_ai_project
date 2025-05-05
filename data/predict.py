import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import logging

# --- Constants ---
IMG_SIZE = 224  # Should match the size used during training
MODEL_PATH = 'C:/neuro_ai_project/models/mri_model.h5'  # Path to your trained model

# --- Set up logging ---
logging.basicConfig(level=logging.ERROR)

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an MRI image for prediction.

    Args:
        image_path (str): Path to the MRI image file.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array, or None on error.
    """
    try:
        # Load the image
        img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading or preprocessing image: {e}")
        return None

def predict_image_class(model, image_path, threshold=0.5):
    """
    Predicts the class of an MRI image using the given model.

    Args:
        model (tf.keras.Model):  The trained Keras model.
        image_path (str): Path to the MRI image file.
        threshold (float): Threshold for binary classification (default: 0.5).

    Returns:
        str: "AD" or "CN", or None on error.
    """
    preprocessed_image = load_and_preprocess_image(image_path)
    if preprocessed_image is None:
        return None  # Error occurred during image loading/preprocessing

    try:
        prediction = model.predict(preprocessed_image)[0][0]  # Get the probability
        if prediction > threshold:
            return "AD"
        else:
            return "CN"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Load the trained model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        exit()  # Exit if the model cannot be loaded

    # Get the image path from the user (replace with your actual image path)
    image_path = input("Enter the path to the MRI image: ")

    # Make a prediction
    predicted_class = predict_image_class(model, image_path)

    if predicted_class:
        print(f"The MRI image is predicted to be: {predicted_class}")
    else:
        print("Error: Could not make a prediction.")