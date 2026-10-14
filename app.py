import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
    layout="centered"
)

# 38 PlantVillage Class Names (Alphabetical order matching training dataset)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to diagnose plant health and diseases using your MobileNetV2 model.")

# Cache the model loading so it downloads from GitHub Release only once
@st.cache_resource
def load_model():
    model_url = "https://github.com/olamify78-prog/plant-disease-app/releases/download/v1.0/plant_disease_model.keras"
    
    with st.spinner("Downloading model from GitHub Release..."):
        model_path = tf.keras.utils.get_file(
            fname="plant_disease_model.keras",
            origin=model_url
        )
            
    model = tf.keras.models.load_model(model_path)
    return model

# Load model safely
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf image..."):
            # Preprocess image to match MobileNetV2 input requirements (224x224)
            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run model prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Map index to readable disease label
            predicted_label = class_names[predicted_class_idx]
            
            # Display formatted results
            st.markdown("---")
            st.subheader("Results")
            st.success(f"**Predicted Disease:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
