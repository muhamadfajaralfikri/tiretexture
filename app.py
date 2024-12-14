import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class names (update as per model training)
class_names = ['normal', 'cracked']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    if img_array.ndim == 2:  # Grayscale image
        img_array = img_array[..., np.newaxis]  # Add channel dimension
    elif img_array.shape[-1] == 3:  # Convert color image to grayscale
        img_array = np.mean(img_array, axis=-1, keepdims=True)  # Average channels
    img_array = img_array.reshape((1, 224, 224, 1))  # Add batch dimension
    return img_array

# Streamlit App
st.title('Tire Texture Classifier')

uploaded_image = st.file_uploader("Upload an image (preferably low resolution)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img, caption="Uploaded Image")

    # Classify button and result display
    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            # Display the prediction
            st.success(f'Prediction: {prediction}')
