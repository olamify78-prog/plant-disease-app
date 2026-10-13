import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload an image of a plant leaf to detect potential diseases using your trained MobileNetV2 model.")

@st.cache_resource
@tf.keras.utils.register_keras_serializable()
def load_my_model():
    with st.spinner("Downloading model from Hugging Face... Please wait (this takes a moment on first boot)."):
        # This automatically handles downloading the actual model binary from your repo
        model_path = hf_hub_download(
            repo_id="kolex12/plant-disease-model",
            filename="plant_disease_model.keras"
        )
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    st.write("Processing image...")
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
