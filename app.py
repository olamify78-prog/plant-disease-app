import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model (cached so it only loads once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

with st.spinner("Loading model... Please wait!"):
    model = load_model()

st.title("🌿 Plant Disease Identification App")
st.write("Upload an image of a plant leaf to check for diseases.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            # Resize image to match what your MobileNetV2 model expects (usually 224x224)
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            st.success(f"Prediction complete! (Confidence: {confidence:.2f}%)")
            st.write(f"Predicted Class Index: {predicted_class}")
