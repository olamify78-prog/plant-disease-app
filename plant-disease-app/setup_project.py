import os

# 1. Create requirements.txt
requirements_content = """streamlit
tensorflow
numpy
pillow
"""
with open("requirements.txt", "w") as f:
    f.write(requirements_content.strip())
print("Created requirements.txt successfully!")

# 2. Create app.py
app_content = '''import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 Plant Disease Identification App")
st.write("Upload a plant leaf image to detect potential diseases using MobileNetV2.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not find or load 'plant_disease_model.h5'. Make sure your model file is in this folder! Error: {e}")

uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Disease"):
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            st.success(f"Prediction complete! Class index: {predicted_class}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
'''

with open("app.py", "w") as f:
    f.write(app_content.strip())
print("Created app.py successfully!")