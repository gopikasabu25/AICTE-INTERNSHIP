import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model.keras")

# Set image size and class names (must match training settings)
IMG_SIZE = (124, 124)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Page setup
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻️ Garbage Material Classifier")
st.markdown("Upload an image of a waste item to classify it into one of six categories.")

# Upload file
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ FIXED


    # Preprocessing
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0  # Normalize to match training
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Debug: Show shape and pixel range
    st.write("🧪 Image shape (should be (1, 124, 124, 3)):", image_array.shape)
    st.write("🧪 Pixel range (should be [0.0, 1.0]):", np.min(image_array), np.max(image_array))

    # Prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Results
    st.markdown(f"### ✅ Predicted Class: **{class_names[predicted_index]}**")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}`")

    # Raw prediction values (optional)
    with st.expander("🔬 See raw model outputs (probabilities for each class)"):
        prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        st.json(prob_dict)

# Footer
st.markdown("---")
st.markdown("Created by Gopika Sabu · Powered by EfficientNetV2B3 and Streamlit")
