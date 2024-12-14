import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Set direktori kerja dan path model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Muat model yang sudah dilatih
model = tf.keras.models.load_model(model_path)

# Tentukan nama kelas (sesuaikan dengan pelatihan model)
class_names = ['normal', 'cracked']

# Muat model deteksi objek (gunakan pretrained MobileNetV2 untuk contoh sederhana)
detection_model = tf.keras.applications.MobileNetV2(weights='imagenet')

def detect_tire(image, confidence_threshold=0.5):
    """
    Fungsi untuk mendeteksi apakah gambar mengandung ban.
    """
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi label menggunakan model MobileNetV2
    predictions = detection_model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)

    # Deteksi objek dengan label relevan "tire" atau "wheel"
    for _, label, confidence in decoded_predictions[0]:
        if ('tire' in label.lower() or 'wheel' in label.lower()) and confidence >= confidence_threshold:
            return True, label, confidence
    return False, None, None

# Fungsi untuk memproses gambar ban
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    if img_array.ndim == 2:  # Grayscale
        img_array = img_array[..., np.newaxis]
    elif img_array.shape[-1] == 3:  # RGB to Grayscale
        img_array = np.mean(img_array, axis=-1, keepdims=True)
    img_array = img_array.reshape((1, 224, 224, 1))
    return img_array

# Aplikasi Streamlit
st.title("Klasifikasi Tekstur Ban (Hanya Bagian Ban)")

uploaded_image = st.file_uploader("Unggah gambar (harus berisi ban)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Buka gambar
    image = Image.open(uploaded_image)

    # Validasi gambar menggunakan deteksi objek
    is_tire, label, confidence = detect_tire(image)
    if not is_tire:
        st.error("Gambar tidak valid: Gambar tidak terdeteksi sebagai bagian ban.")
    else:
        st.success(f"Bagian ban terdeteksi dengan label: '{label}' (kepercayaan: {confidence:.2f}).")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)

        with col2:
            if st.button("Klasifikasi Tekstur Ban"):
                try:
                    img_array = preprocess_image(image)
                    result = model.predict(img_array)

                    if result.shape[-1] == 1:
                        result = np.hstack([1 - result, result])
                    predicted_class = np.argmax(result)
                    prediction = class_names[predicted_class]

                    st.success(f"Prediksi Tekstur: {prediction}")

                    normal_conf = result[0][0] * 100
                    cracked_conf = result[0][1] * 100
                    st.markdown(f"Kepercayaan **Normal**: {normal_conf:.2f}%")
                    st.markdown(f"Kepercayaan **Cracked**: {cracked_conf:.2f}%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
