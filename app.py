import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Check if model exists
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure 'cnn_model_cracked.keras' is in the working directory.")
    st.stop()

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class names (update as per model training)
class_names = ['normal', 'cracked']

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize to model's expected input size
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  # Normalize pixel values

    if img_array.ndim == 2:  # Grayscale image
        img_array = img_array[..., np.newaxis]  # Add channel dimension
        img_array = np.repeat(img_array, 3, axis=-1)  # Repeat channels for RGB
    elif img_array.shape[-1] != 3:  # Ensure RGB input
        st.error("Unsupported image format. Please upload a valid RGB or grayscale image.")
        st.stop()
    
    # Reshape to add batch dimension
    img_array = img_array.reshape((1, 224, 224, 3))  
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
        resized_img = image.resize((224, 224))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    # Classify button and result display
    with col2:
        if st.button('Classify'):
            try:
                # Preprocess the uploaded image
                img_array = preprocess_image(image)

                # Make a prediction using the pre-trained model
                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                # Display the prediction
                if prediction == 'cracked':
                    st.warning("Warning: The tire may be damaged! Please inspect it further for safety.")
                else:
                    st.success("The tire texture appears to be normal.")

                st.write(f'Prediction Confidence: {result[0][predicted_class]:.2f}')

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
