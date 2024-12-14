import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set direktori kerja dan path model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Muat model yang sudah dilatih
model = tf.keras.models.load_model(model_path)

# Tentukan nama kelas (sesuaikan dengan pelatihan model)
class_names = ['normal', 'cracked']

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    # Ubah ukuran gambar untuk mencocokkan input model
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  # Normalisasi nilai piksel
    
    # Periksa apakah gambar grayscale atau berwarna
    if img_array.ndim == 2:  # Jika gambar grayscale
        img_array = img_array[..., np.newaxis]  # Tambahkan dimensi channel
    elif img_array.shape[-1] == 3:  # Jika gambar berwarna (RGB), konversi ke grayscale
        img_array = np.mean(img_array, axis=-1, keepdims=True)  # Menghitung rata-rata untuk menghasilkan 1 channel
    
    # Ubah bentuk menjadi format yang dibutuhkan model
    img_array = img_array.reshape((1, 224, 224, 1))  # Tambahkan dimensi batch dan pastikan 1 channel
    return img_array

# Aplikasi Streamlit
st.title('Klasifikasi Tekstur Ban')

uploaded_image = st.file_uploader("Unggah gambar (lebih baik resolusi rendah)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Buka gambar yang diunggah
    image = Image.open(uploaded_image)

    # Tampilkan gambar yang diunggah
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((224, 224))  # Tampilkan versi yang lebih besar
        st.image(resized_img, caption="Gambar yang Diupload", use_column_width=True)

    # Tombol klasifikasi dan tampilan hasil
    with col2:
        if st.button('Klasifikasi'):
            try:
                # Proses gambar yang diunggah
                img_array = preprocess_image(image)

                # Lakukan prediksi menggunakan model yang telah dilatih
                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                # Tampilkan hasil prediksi
                st.success(f'Prediksi: {prediction}')
                st.write(f"Kepercayaan: {result[0][predicted_class]:.2f}")

            except ValueError as ve:
                st.error(f"Terjadi kesalahan pada gambar: {ve}")
            except Exception as e:
                st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
