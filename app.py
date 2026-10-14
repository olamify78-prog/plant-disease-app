import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection App")

MODEL_URL = "https://github.com/olamify78-prog/plant-disease-app/releases/download/v1.0/plant_disease_model.keras"
MODEL_PATH = os.path.abspath("plant_disease_model.keras")

@st.cache_resource
@tf.keras.utils.register_keras_serializable()
def load_my_model():
    # Download if file doesn't exist or is too small (< 500 KB)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 500000:
        with st.spinner("Downloading model from GitHub Releases... Please wait."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Download exception: {e}")
                return None

    # Verify if the file is a valid ZIP archive (.keras files are ZIPs)
    try:
        with open(MODEL_PATH, "rb") as f:
            header = f.read(4)
        
        # ZIP magic bytes are b'PK\x03\x04'
        if header != b'PK\x03\x04':
            with open(MODEL_PATH, "r", errors="ignore") as f:
                content = f.read(400)
            st.error(f"**Invalid Model File Downloaded:** GitHub returned a non-ZIP file (like a notebook or HTML page).\n\nContent preview:\n`{content}`")
            os.remove(MODEL_PATH)
            return None
    except Exception as e:
        st.error(f"Error checking file format: {e}")
        return None

    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
    if model is not None:
        st.success("Model loaded successfully!")
    else:
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
            
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    if st.button('Predict Disease'):
        with st.spinner('Analyzing leaf...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            st.markdown("### Results")
            st.write(f"**Predicted Class Index:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
