import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

st.title("🌱 Plant Disease Detection")
st.markdown("Upload a leaf image and get predictions from a pretrained model.")

# ✅ Load model from TFHub
@st.cache_resource
def load_model():
    return hub.load("https://www.kaggle.com/models/rishitdagli/plant-disease/TensorFlow2/plant-disease/1")

model = load_model()

# ✅ Class labels (38 classes from PlantVillage)
class_names = [
    "Apple → Apple scab", "Apple → Black rot", "Apple → Cedar apple rust", "Apple → healthy",
    "Blueberry → healthy", "Cherry → Powdery mildew", "Cherry → healthy",
    "Corn → Cercospora leaf spot Gray leaf spot", "Corn → Common rust",
    "Corn → Northern Leaf Blight", "Corn → healthy", "Grape → Black rot",
    "Grape → Esca (Black Measles)", "Grape → Leaf blight", "Grape → healthy",
    "Orange → Citrus greening", "Peach → Bacterial spot", "Peach → healthy",
    "Pepper → Bacterial spot", "Pepper → healthy", "Potato → Early blight",
    "Potato → Late blight", "Potato → healthy", "Raspberry → healthy", "Soybean → healthy",
    "Squash → Powdery mildew", "Strawberry → Leaf scorch", "Strawberry → healthy",
    "Tomato → Bacterial spot", "Tomato → Early blight", "Tomato → Late blight",
    "Tomato → Leaf Mold", "Tomato → Septoria leaf spot", "Tomato → Spider mites",
    "Tomato → Target Spot", "Tomato → Yellow Leaf Curl Virus", "Tomato → Mosaic virus",
    "Tomato → healthy"
]

# ✅ Image preprocessing
def preprocess_image(uploaded_file, size=(224, 224)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(size)
    img_array = np.array(img) / 255.0
    tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return tf.expand_dims(tensor, 0)  # Add batch dimension

# ✅ Upload image
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    with st.spinner("Predicting..."):
        input_tensor = preprocess_image(uploaded_file)
        predictions = model(input_tensor)[0].numpy()

        # Top-k results
        top_k = 3
        top_indices = predictions.argsort()[-top_k:][::-1]

        st.subheader("🔍 Top Predictions:")
        for i in top_indices:
            confidence = predictions[i] * 100
            st.write(f"**{class_names[i]}** — {confidence:.2f}%")

        st.markdown("---")

        if predictions[top_indices[0]] < 0.5:
            st.warning("⚠️ Low confidence prediction. Try a clearer or closer image.")
