import os
import gdown
import streamlit as st
import tensorflow as tf

MODEL_PATH = "plant_disease_model.keras"
FILE_ID = "1ZwezuGGoGzQxXJDoMsKGbxzNO_jISGXS"

@st.cache_resource
@tf.keras.utils.register_keras_serializable()
def load_my_model():
    # If the file doesn't exist locally in the container, download it using the file ID
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            
    # Load the model safely
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
