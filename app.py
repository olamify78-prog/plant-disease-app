import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection App")

# Your verified GitHub Release direct download link
MODEL_URL = "https://github.com/olamify78-prog/plant-disease-app/releases/download/v1.0/plant_disease_model.keras"
MODEL_PATH = "plant_disease_model.keras"

@st.cache_resource
@tf.keras.utils.register_keras_serializable()
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Downloading model from GitHub Releases... Please wait."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
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
