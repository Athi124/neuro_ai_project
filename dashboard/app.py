# dashboard/app.py
import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pickle
import os
from PIL import Image  # type: ignore
import io
import logging

base_data_dir = "C:/neuro_ai_project/data/"
mri_dir = os.path.join(base_data_dir, "mri")
genomics_dir = os.path.join(base_data_dir, "genomics")
clinical_dir = os.path.join(base_data_dir, "clinical")
models_dir = "C:/neuro_ai_project/models/"


# Placeholder for loading trained models
def load_model(filepath):
    st.info(f"Placeholder: Loading model from {filepath}")
    return None

# Placeholder for making MRI predictions
def predict_mri(image_data, model):
    st.info("Placeholder: Making MRI predictions")
    num_samples = image_data.shape[0] if len(image_data.shape) > 0 else 5
    return np.random.rand(num_samples, 2)

# Placeholder for making clinical/genomic predictions
def predict_cg(data, model):
    st.info("Placeholder: Making clinical/genomic predictions")
    num_samples = data.shape[0] if len(data.shape) > 0 else 5
    return np.random.rand(num_samples, 2)

if __name__ == "__main__":
    st.title("AI-Powered Neurodegenerative Disease Prediction (Demonstration)")
    st.subheader("Demonstration Dashboard")
    st.warning(
        "Encountered issues with training the AI models. Results shown are based on placeholder predictions."
    )

    # Load a small subset of simulated data for display
    try:
        genomics_df_display = (
            pd.read_csv(os.path.join(genomics_dir, "genomics_data.csv")).head(5)
        )
        clinical_df_display = (
            pd.read_csv(os.path.join(clinical_dir, "clinical_data.csv")).head(5)
        )
        mri_images_display = np.load(os.path.join(mri_dir, "images.npy"))[:5]
    except FileNotFoundError:
        st.error(
            "Error: Simulated data files not found. Please run simulate_data.py first."
        )
        st.stop()

    st.subheader("Simulated Input Data (First 5 Samples)")

    st.subheader("Simulated Genomics Data:")
    st.dataframe(genomics_df_display.drop("label", axis=1))

    st.subheader("Simulated Clinical Data:")
    st.dataframe(clinical_df_display.drop("label", axis=1))

    st.subheader("Placeholder for MRI Data:")
    st.info(
        "Due to issues encountered with model training, the MRI data processing and display are placeholders in this demonstration."
    )
    # You could add a placeholder image or some descriptive text here if you like

    if st.button("Run Prediction on Displayed Samples"):
        # Placeholder loading of models
        mri_model = load_model(os.path.join(models_dir, "mri_model.h5"))
        cg_model = load_model(os.path.join(models_dir, "cg_model.pkl"))

        # Placeholder predictions on the displayed data
        mri_predictions = predict_mri(mri_images_display, mri_model)
        cg_predictions = predict_cg(
            pd.concat(
                [
                    clinical_df_display.drop("label", axis=1),
                    genomics_df_display.drop("label", axis=1),
                ],
                axis=1,
            ),
            cg_model,
        )

        # Simple averaging of placeholder predictions
        integrated_predictions = (mri_predictions + cg_predictions) / 2
        final_predictions = np.argmax(integrated_predictions, axis=1)

        st.subheader("Placeholder Prediction Results (First 5 Samples)")
        results_df = pd.DataFrame(
            {
                "Placeholder MRI Probabilities (Class 0, Class 1)": [
                    f"{p[0]:.2f}, {p[1]:.2f}" for p in mri_predictions
                ],
                "Placeholder Clinical/Genomic Probabilities (Class 0, Class 1)": [
                    f"{p[0]:.2f}, {p[1]:.2f}" for p in cg_predictions
                ],
                "Placeholder Integrated Prediction (Class)": final_predictions,
            }
        )
        st.dataframe(results_df)

    st.subheader("Explanation:")
    st.write(
        "This demonstration showcases the concept of using multimodal data for neurodegenerative disease prediction. Due to technical challenges encountered, the AI model training was not fully successful, and the prediction results shown are based on placeholder logic. In a fully functional system, trained AI models would process the MRI, genomics, and clinical data to provide more meaningful predictions."
    )

    # --- Displaying your healthcare_dataset.csv ---
    st.header("Displaying Healthcare Dataset")
    healthcare_csv_path = "C:/Users/Akash/Downloads/archive/healthcare_dataset.csv"
    if os.path.exists(healthcare_csv_path):
        try:
            healthcare_data = pd.read_csv(healthcare_csv_path)
            st.dataframe(healthcare_data)
        except Exception as e:
            st.error(f"Error loading healthcare dataset: {e}")
    else:
        st.warning(f"Healthcare dataset not found at: {healthcare_csv_path}")

    # --- Displaying the user provided MRI image ---
    st.header("Displaying User Provided MRI Image")  # Changed title
    user_mri_image_path = "C:/Users/Akash/Downloads/29 no.jpg"
    if os.path.exists(user_mri_image_path):
        try:
            image = Image.open(user_mri_image_path)
            st.image(
                image,
                caption="Uploaded MRI Image",
                use_container_width=True,
            )  # Changed caption
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.warning(f"Image not found at: {user_mri_image_path}")

    # --- Image Upload ---
    st.header("Image Upload")
    image_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "dcm"]
    )  # Add "dcm"
    if image_file is not None:
        try:
            if image_file.name.lower().endswith(".dcm"):
                try:
                    import pydicom  # Import pydicom here, only when needed
                except ImportError:
                    st.error(
                        "pydicom is required to handle DICOM files.  Please install it with: pip install pydicom"
                    )
                    st.stop()
                dicom_data = pydicom.dcmread(image_file)
                img_array = dicom_data.pixel_array
                if len(img_array.shape) == 2:  # grayscale
                    img = Image.fromarray(img_array.astype(np.uint8))
                    st.image(
                        img,
                        caption="Uploaded DICOM Image",
                        use_container_width=True,
                    )
                elif (
                    len(img_array.shape) == 3 and img_array.shape[2] == 3
                ):  # RGB
                    img = Image.fromarray(img_array.astype(np.uint8), "RGB")
                    st.image(
                        img,
                        caption="Uploaded DICOM Image",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "DICOM image format not handled for display.  Provide a grayscale or RGB image."
                    )
            else:
                image = Image.open(image_file)
                st.image(
                    image, caption="Uploaded Image", use_container_width=True
                )
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # --- Dataset Upload (CSV, Excel) ---
    st.header("Dataset Upload")
    data_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    if data_file is not None:
        try:
            if data_file.name.lower().endswith(".csv"):
                data = pd.read_csv(data_file)
            elif data_file.name.lower().endswith(".xlsx"):
                data = pd.read_excel(data_file)
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # Add other Streamlit elements here...
    st.write("This is some text in my app.")
    # --- Check for TensorFlow and install if not found ---
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        st.error(
            "TensorFlow is not installed. Please install it. "
            "You can try: `pip install tensorflow` in your terminal."
        )
        st.stop()  # Stop execution if TensorFlow is missing

    import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import logging
import streamlit as st
from PIL import Image
import io
import os
import pandas as pd
import random

# --- Constants ---
IMG_SIZE = 224  # Should match the size used during training
MODEL_PATH = 'C:/neuro_ai_project/models/mri_model.h5'  # Path to your trained model
DEFAULT_IMAGE = 'C:/neuro_ai_project/data/Train/AD/ad_image1.jpg'  # Corrected default image path
BASE_DIR = "C:/neuro_ai_project/"

# --- Set up logging ---
logging.basicConfig(level=logging.ERROR)


# --- Load the model ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None


# --- Preprocess Image ---
def preprocess_image(image_file):
    """
    Loads and preprocesses an MRI image from a file-like object.
    Args:
        image_file (streamlit.UploadedFile): Uploaded file
    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array, or None on error.
    """
    try:
        # Read the image data from the file
        img = Image.open(image_file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading or preprocessing image: {e}")
        return None


# --- Prediction ---
def predict_image_class(model, image_file, threshold=0.5):
    """
    Predicts the class of an MRI image using the given model.
    Args:
        model (tf.keras.Model): The trained Keras model.
        image_file (streamlit.UploadedFile): Uploaded file
        threshold (float): Threshold for binary classification (default: 0.5).
    Returns:
        str: "AD" or "CN", or None on error.
    """
    preprocessed_image = preprocess_image(image_file)
    if preprocessed_image is None:
        return None  # Error occurred during image loading/preprocessing
    try:
        prediction = model.predict(preprocessed_image)[0][0]  # Get the probability
        if prediction > threshold:
            return "Alzheimer's Prediction"
        else:
            return "Cognitive Normal"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None


# --- Main App ---
def main():
    """
    Main function for the Streamlit web application.
    """
    st.title("MRI Image Classification and Data Display")

    # Load the model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model path.")
        return  # Stop if the model isn't loaded

    # File uploader
    image_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

    # CSV File Uploader
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if image_file is not None:
        # Display the image
        st.image(image_file, caption="Uploaded MRI Image", use_column_width=True)
        # Make a prediction
        predicted_class = predict_image_class(model, image_file)
        if predicted_class:
            st.write(f"The MRI image is predicted to be: **{predicted_class}**")
        else:
            st.error("Could not make a prediction. Please check the image and try again.")
    else:
        # display default image
        if os.path.exists(DEFAULT_IMAGE): # Check if the default image exists
            st.image(DEFAULT_IMAGE, caption="Default MRI Image", use_column_width=True)
        else:
            st.error(f"Default image not found at {DEFAULT_IMAGE}. Please check the path.")

    if csv_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Display the CSV data
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading or displaying CSV file: {e}")

    # Generate and display dummy Alzheimer's data
    st.header("Alzheimer's Disease Information")
    st.write("Here's some dummy data related to Alzheimer's Disease:")

    # Define dummy data
    alzheimer_types = ["Sporadic Alzheimer's Disease", "Familial Alzheimer's Disease",
                      "Early-Onset Alzheimer's Disease", "Late-Onset Alzheimer's Disease"]
    alzheimer_stages = ["Preclinical Stage", "Mild Cognitive Impairment (MCI)",
                        "Mild Dementia", "Moderate Dementia", "Severe Dementia"]
    risk_factors = ["Age", "Genetics", "Family History",
                    "Cardiovascular Health", "Lifestyle Factors"]
    symptoms = ["Memory Loss", "Cognitive Decline",
                "Behavioral Changes", "Language Problems", "Disorientation"]
    diagnosis_methods = ["Medical History", "Cognitive Tests",
                        "Brain Imaging (MRI, PET)", "Biomarkers (CSF, Blood)"]

    # Use random.choice to select a random entry from the lists
    random_type = random.choice(alzheimer_types)
    random_stage = random.choice(alzheimer_stages)
    random_risk_factor = random.choice(risk_factors)
    random_symptom = random.choice(symptoms)
    random_diagnosis = random.choice(diagnosis_methods)

    # create a dictionary
    dummy_data = {
        "Type of Alzheimer's": random_type,
        "Stage of Alzheimer's": random_stage,
        "Key Risk Factor": random_risk_factor,
        "Common Symptom": random_symptom,
        "Diagnosis Method": random_diagnosis
    }

    # Display the dummy data as a dictionary
    st.write(dummy_data)

    # Display the data in a formatted way
    st.subheader("Random Alzheimer's Facts")
    st.write(f"Type: {random_type}")
    st.write(f"Stage: {random_stage}")
    st.write(f"Risk Factor: {random_risk_factor}")
    st.write(f"Symptom: {random_symptom}")
    st.write(f"Diagnosis: {random_diagnosis}")


if __name__ == "__main__":
    main()
    # --- Constants ---
IMG_SIZE = 224  # Should match the size used during training
MODEL_PATH = 'C:/neuro_ai_project/models/mri_model.h5'  # Path to your trained model
DEFAULT_IMAGE = 'C:/neuro_ai_project/data/Train/AD/ad_image1.jpg'  # Corrected default image path
BASE_DIR = "C:/neuro_ai_project/"

# --- Set up logging ---
logging.basicConfig(level=logging.ERROR)


# --- Load the model ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None


# --- Preprocess Image ---
def preprocess_image(image_file):
    """
    Loads and preprocesses an MRI image from a file-like object.
    Args:
        image_file (streamlit.UploadedFile): Uploaded file
    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array, or None on error.
    """
    try:
        # Read the image data from the file
        img = Image.open(image_file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading or preprocessing image: {e}")
        return None


# --- Prediction ---
def predict_image_class(model, image_file, threshold=0.5):
    """
    Predicts the class of an MRI image using the given model.
    Args:
        model (tf.keras.Model): The trained Keras model.
        image_file (streamlit.UploadedFile): Uploaded file
        threshold (float): Threshold for binary classification (default: 0.5).
    Returns:
        str: "AD" or "CN", or None on error.
    """
    preprocessed_image = preprocess_image(image_file)
    if preprocessed_image is None:
        return None  # Error occurred during image loading/preprocessing
    try:
        prediction = model.predict(preprocessed_image)[0][0]  # Get the probability
        if prediction > threshold:
            return "Alzheimer's Prediction"
        else:
            return "Cognitive Normal"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None


# --- Main App ---
def main():
    """
    Main function for the Streamlit web application.
    """

    # --- Add a header with a logo ---
    col1, col2 = st.columns([1, 4])  # Adjust column widths as needed
    with col1:
        # Replace with the path to your logo file.  Make sure this path is correct
        logo_path = "C:/neuro_ai_project/dashboard/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)  # Adjust the width as needed
        else:
            st.error(f"Logo image not found at {logo_path}")
    with col2:
        st.title("MRI Image Analysis and Data Hub", )
        st.markdown(
            "<h2 style='color: #4CAF50;'>Empowering Alzheimer's Research</h2>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(135deg, #e0f7fa, #c2e59d);
        }
        .main h1, .main h2, .main h3, .main h4 {
            color: #0d47a1;
        }
        .stButton>button {
            color: #fff;
            background-color: #4CAF50;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #388e3c;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stFileUploader label {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stFileUploader label:hover {
            background-color: #388e3c;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stDataFrame {
             border: 1px solid #e0e0e0;
             border-radius: 5px;
             padding: 10px;
             background-color: #f5f5f5;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load the model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model path.")
        return  # Stop if the model isn't loaded

    # File uploader
    image_file = st.file_uploader(
        "Upload an MRI image for analysis", type=["png", "jpg", "jpeg"]
    )

    # CSV File Uploader
    csv_file = st.file_uploader(
        "Upload a CSV file to display", type=["csv"]
    )

    if image_file is not None:
        # Display the image
        st.image(image_file, caption="Uploaded MRI Image", use_column_width=True)
        # Make a prediction
        predicted_class = predict_image_class(model, image_file)
        if predicted_class:
            st.success(f"The MRI image is predicted to be: **{predicted_class}**")
        else:
            st.error(
                "Could not make a prediction. Please check the image and try again."
            )
    else:
        # display default image
        if os.path.exists(DEFAULT_IMAGE):  # Check if the default image exists
            st.image(
                DEFAULT_IMAGE, caption="Default MRI Image", use_column_width=True
            )
        else:
            st.error(
                f"Default image not found at {DEFAULT_IMAGE}. Please check the path."
            )

    if csv_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Display the CSV data
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading or displaying CSV file: {e}")

    # Generate and display dummy Alzheimer's data
    st.header("Alzheimer's Disease Insights")
    st.markdown(
        """
        <p style='font-size: 18px; color: #555;'>
            Here's a glimpse into Alzheimer's Disease:
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Define dummy data
    alzheimer_types = [
        "Sporadic Alzheimer's Disease",
        "Familial Alzheimer's Disease",
        "Early-Onset Alzheimer's Disease",
        "Late-Onset Alzheimer's Disease",
    ]
    alzheimer_stages = [
        "Preclinical Stage",
        "Mild Cognitive Impairment (MCI)",
        "Mild Dementia",
        "Moderate Dementia",
        "Severe Dementia",
    ]
    risk_factors = [
        "Age",
        "Genetics",
        "Family History",
        "Cardiovascular Health",
        "Lifestyle Factors",
    ]
    symptoms = [
        "Memory Loss",
        "Cognitive Decline",
        "Behavioral Changes",
        "Language Problems",
        "Disorientation",
    ]
    diagnosis_methods = [
        "Medical History",
        "Cognitive Tests",
        "Brain Imaging (MRI, PET)",
        "Biomarkers (CSF, Blood)",
    ]

    # Use random.choice to select a random entry from the lists
    random_type = random.choice(alzheimer_types)
    random_stage = random.choice(alzheimer_stages)
    random_risk_factor = random.choice(risk_factors)
    random_symptom = random.choice(symptoms)
    random_diagnosis = random.choice(diagnosis_methods)

    # create a dictionary
    dummy_data = {
        "Type of Alzheimer's": random_type,
        "Stage of Alzheimer's": random_stage,
        "Key Risk Factor": random_risk_factor,
        "Common Symptom": random_symptom,
        "Diagnosis Method": random_diagnosis,
    }

    # Display the dummy data as a dictionary
    st.write(dummy_data)

    # Display the data in a formatted way
    st.subheader("Random Alzheimer's Facts")
    st.markdown(
        f"""
        <ul style='list-style-type: disc; font-size: 16px; padding-left: 20px;'>
            <li><strong>Type:</strong> <span style='color: #1a5276;'>{random_type}</span></li>
            <li><strong>Stage:</strong> <span style='color: #1a5276;'>{random_stage}</span></li>
            <li><strong>Risk Factor:</strong> <span style='color: #1a5276;'>{random_risk_factor}</span></li>
            <li><strong>Symptom:</strong> <span style='color: #1a5276;'>{random_symptom}</span></li>
            <li><strong>Diagnosis:</strong> <span style='color: #1a5276;'>{random_diagnosis}</span></li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    # --- Constants ---
IMG_SIZE = 224  # Should match the size used during training
MODEL_PATH = 'C:/neuro_ai_project/models/mri_model.h5'  # Path to your trained model
DEFAULT_IMAGE = 'C:/neuro_ai_project/data/Train/AD/ad_image1.jpg'  # Corrected default image path
BASE_DIR = "C:/neuro_ai_project/"

# --- Set up logging ---
logging.basicConfig(level=logging.ERROR)


# --- Load the model ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None


# --- Preprocess Image ---
def preprocess_image(image_file):
    """
    Loads and preprocesses an MRI image from a file-like object.
    Args:
        image_file (streamlit.UploadedFile): Uploaded file
    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array, or None on error.
    """
    try:
        # Read the image data from the file
        img = Image.open(image_file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading or preprocessing image: {e}")
        return None


# --- Prediction ---
def predict_image_class(model, image_file, threshold=0.5):
    """
    Predicts the class of an MRI image using the given model.
    Args:
        model (tf.keras.Model): The trained Keras model.
        image_file (streamlit.UploadedFile): Uploaded file
        threshold (float): Threshold for binary classification (default: 0.5).
    Returns:
        str: "AD" or "CN", or None on error.
    """
    preprocessed_image = preprocess_image(image_file)
    if preprocessed_image is None:
        return None  # Error occurred during image loading/preprocessing
    try:
        prediction = model.predict(preprocessed_image)[0][0]  # Get the probability
        if prediction > threshold:
            return "Alzheimer's Prediction"
        else:
            return "Cognitive Normal"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None


# --- Main App ---
def main():
    """
    Main function for the Streamlit web application.
    """

    # --- Add a header with a logo ---
    col1, col2 = st.columns([1, 4])  # Adjust column widths as needed
    with col1:
        # Replace with the path to your logo file.  Make sure this path is correct
        logo_path = "C:/neuro_ai_project/dashboard/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)  # Adjust the width as needed
        else:
            st.error(f"Logo image not found at {logo_path}")
    with col2:
        st.title("MRI Image Analysis and Data Hub", )
        st.markdown(
            "<h2 style='color: #4CAF50;'>Empowering Alzheimer's Research</h2>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(135deg, #e0f7fa, #c2e59d);
        }
        .main h1, .main h2, .main h3, .main h4 {
            color: #0d47a1;
        }
        .stButton>button {
            color: #fff;
            background-color: #4CAF50;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #388e3c;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stFileUploader label {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stFileUploader label:hover {
            background-color: #388e3c;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stDataFrame {
             border: 1px solid #e0e0e0;
             border-radius: 5px;
             padding: 10px;
             background-color: #f5f5f5;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load the model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model path.")
        return  # Stop if the model isn't loaded

    # File uploader
    image_file = st.file_uploader(
        "Upload an MRI image for analysis", type=["png", "jpg", "jpeg"]
    )

    # CSV File Uploader
    csv_file = st.file_uploader(
        "Upload a CSV file to display", type=["csv"]
    )

    if image_file is not None:
        # Display the image
        st.image(image_file, caption="Uploaded MRI Image", use_column_width=True)
        # Make a prediction
        predicted_class = predict_image_class(model, image_file)
        if predicted_class:
            st.success(f"The MRI image is predicted to be: **{predicted_class}**")
        else:
            st.error(
                "Could not make a prediction. Please check the image and try again."
            )
    else:
        # display default image
        if os.path.exists(DEFAULT_IMAGE):  # Check if the default image exists
            st.image(
                DEFAULT_IMAGE, caption="Default MRI Image", use_column_width=True
            )
        else:
            st.error(
                f"Default image not found at {DEFAULT_IMAGE}. Please check the path."
            )

    if csv_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Display the CSV data
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading or displaying CSV file: {e}")

    # Generate and display dummy Alzheimer's data
    st.header("Alzheimer's Disease Insights")
    st.markdown(
        """
        <p style='font-size: 18px; color: #555;'>
            Here's a glimpse into Alzheimer's Disease:
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Define dummy data
    alzheimer_types = [
        "Sporadic Alzheimer's Disease",
        "Familial Alzheimer's Disease",
        "Early-Onset Alzheimer's Disease",
        "Late-Onset Alzheimer's Disease",
    ]
    alzheimer_stages = [
        "Preclinical Stage",
        "Mild Cognitive Impairment (MCI)",
        "Mild Dementia",
        "Moderate Dementia",
        "Severe Dementia",
    ]
    risk_factors = [
        "Age",
        "Genetics",
        "Family History",
        "Cardiovascular Health",
        "Lifestyle Factors",
    ]
    symptoms = [
        "Memory Loss",
        "Cognitive Decline",
        "Behavioral Changes",
        "Language Problems",
        "Disorientation",
    ]
    diagnosis_methods = [
        "Medical History",
        "Cognitive Tests",
        "Brain Imaging (MRI, PET)",
        "Biomarkers (CSF, Blood)",
    ]

    # Use random.choice to select a random entry from the lists
    random_type = random.choice(alzheimer_types)
    random_stage = random.choice(alzheimer_stages)
    random_risk_factor = random.choice(risk_factors)
    random_symptom = random.choice(symptoms)
    random_diagnosis = random.choice(diagnosis_methods)

    # create a dictionary
    dummy_data = {
        "Type of Alzheimer's": random_type,
        "Stage of Alzheimer's": random_stage,
        "Key Risk Factor": random_risk_factor,
        "Common Symptom": random_symptom,
        "Diagnosis Method": random_diagnosis,
    }

    # Display the dummy data as a dictionary
    st.write(dummy_data)

    # Display the data in a formatted way
    st.subheader("Random Alzheimer's Facts")
    st.markdown(
        f"""
        <ul style='list-style-type: disc; font-size: 16px; padding-left: 20px;'>
            <li><strong>Type:</strong> <span style='color: #1a5276;'>{random_type}</span></li>
            <li><strong>Stage:</strong> <span style='color: #1a5276;'>{random_stage}</span>
                 Alzheimer's progresses through several stages, each with distinct characteristics. The Preclinical stage is marked by changes in the brain, but without noticeable symptoms.
                 Mild Cognitive Impairment (MCI) involves slight cognitive decline, while mild dementia brings about more pronounced memory loss and confusion.
                 As the disease advances to moderate dementia, individuals experience significant difficulties with daily activities, and severe dementia leads to a loss of speech and ultimately dependence on others for care.
                 Progression through these stages varies among individuals.
                 </li>
            <li><strong>Risk Factor:</strong> <span style='color: #1a5276;'>{random_risk_factor}</span></li>
            <li><strong>Symptom:</strong> <span style='color: #1a5276;'>{random_symptom}</span></li>
            <li><strong>Diagnosis:</strong> <span style='color: #1a5276;'>{random_diagnosis}</span>
                 Diagnosing Alzheimer's disease involves a comprehensive approach. Initially, a thorough medical history is taken, and cognitive tests are administered to evaluate memory and thinking skills.
                 Brain imaging techniques such as MRI and PET scans can help detect changes in brain structure and function.
                 Additionally, the analysis of biomarkers in cerebrospinal fluid (CSF) and blood samples is increasingly used to identify specific proteins associated with Alzheimer's.
                 Genetic testing may be conducted in certain cases, particularly when familial Alzheimer's is suspected.
                 Early and accurate diagnosis is crucial for effective management and intervention.
                 </li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    

