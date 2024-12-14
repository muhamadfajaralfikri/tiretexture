import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import traceback

# Set direktori kerja dan path model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Muat model yang sudah dilatih
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file model tersedia di direktori.")
    st.text(traceback.format_exc())
    st.stop()

# Tentukan nama kelas (sesuaikan dengan pelatihan model)
class_names = ['normal', 'cracked']

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    img = image.resize((224, 224))  # Ubah ukuran gambar
    img_array = np.array(img) / 255.0  # Normalisasi
    
    # Pastikan input memiliki 3 channel (RGB)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)  # Ubah grayscale jadi RGB
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    # Tambahkan batch dimension
    img_array = img_array.reshape((1, 224, 224, 3))
    return img_array

# Aplikasi Streamlit
st.title('Klasifikasi Tekstur Ban')

uploaded_image = st.file_uploader("Unggah gambar (lebih baik resolusi rendah)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Tampilkan gambar yang diunggah
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    if st.button('Klasifikasi'):
        try:
            # Proses gambar
            img_array = preprocess_image(image)

            # Prediksi
            with st.spinner("Model sedang memproses gambar..."):
                result = model.predict(img_array)
            
            # Debugging tambahan
            st.text(f"Raw Output Model: {result}")
            
            if result.shape[-1] == 1:  # Binary classification
                result = np.hstack([1 - result, result])
            
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediksi: {prediction}')
            
            normal_confidence = result[0][0] * 100
            cracked_confidence = result[0][1] * 100

            st.write(f"Kepercayaan Normal: {normal_confidence:.2f}%")
            st.write(f"Kepercayaan Cracked: {cracked_confidence:.2f}%")

        except Exception as e:
            st.error(f"Kesalahan tidak terduga: {e}")
            st.text(traceback.format_exc())
