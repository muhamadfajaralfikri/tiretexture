import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Jika model menerima gambar RGB, pastikan gambar tetap dalam format RGB
    if img_array.ndim == 2:  # Jika gambar grayscale
        img_array = np.stack([img_array]*3, axis=-1)  # Mengulang saluran untuk menjadikannya RGB
    
    # Ubah bentuk menjadi format yang dibutuhkan model
    img_array = img_array.reshape((1, 224, 224, 3))  # Tambahkan dimensi batch
    return img_array

# Fungsi untuk memotong bagian ban (contoh sederhana)
def crop_tire(image):
    # Misalnya, crop bagian tengah gambar yang berukuran 200x200 piksel
    width, height = image.size
    left = width // 4
    top = height // 4
    right = width * 3 // 4
    bottom = height * 3 // 4
    
    # Potong gambar untuk hanya mencakup area ban
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

# Fungsi untuk menampilkan grafik probabilitas
def plot_confidence_chart(normal_confidence, cracked_confidence):
    categories = ['Normal', 'Cracked']
    probabilities = [normal_confidence, cracked_confidence]
    
    fig, ax = plt.subplots()
    ax.bar(categories, probabilities, color=['green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence for Each Class')

    st.pyplot(fig)

# Aplikasi Streamlit
st.title('Klasifikasi Tekstur Ban')

uploaded_image = st.file_uploader("Unggah gambar (lebih baik resolusi rendah)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Buka gambar yang diunggah
    image = Image.open(uploaded_image)

    # Potong gambar untuk hanya mencakup area ban
    cropped_image = crop_tire(image)

    # Tampilkan gambar yang dipotong
    col1, col2 = st.columns(2)

    with col1:
        resized_img = cropped_image.resize((224, 224))  # Tampilkan versi yang lebih besar
        st.image(resized_img, caption="Gambar Bagian Ban", use_column_width=True)

    # Tombol klasifikasi dan tampilan hasil
    with col2:
        if st.button('Klasifikasi'):
            try:
                # Proses gambar yang dipotong
                img_array = preprocess_image(cropped_image)

                # Lakukan prediksi menggunakan model yang telah dilatih
                result = model.predict(img_array)
                
                # Periksa bentuk output model
                if result.shape[-1] == 1:  # Jika model hanya mengeluarkan satu output (binary classification)
                    result = np.hstack([1 - result, result])  # Buat dua kelas (normal vs cracked)
                
                # Ambil kelas dengan probabilitas tertinggi
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                # Tampilkan hasil prediksi
                st.success(f'Prediksi: {prediction}')
                
                # Tampilkan kepercayaan atau probabilitas untuk setiap kelas
                normal_confidence = result[0][0] * 100
                cracked_confidence = result[0][1] * 100

                # Menampilkan kepercayaan dengan warna yang sesuai
                if prediction == 'normal':
                    st.markdown(f"<h3 style='color:green;'>Kepercayaan untuk 'normal': {normal_confidence:.2f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:red;'>Kepercayaan untuk 'cracked': {cracked_confidence:.2f}%</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:red;'>Kepercayaan untuk 'normal': {normal_confidence:.2f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:green;'>Kepercayaan untuk 'cracked': {cracked_confidence:.2f}%</h3>", unsafe_allow_html=True)

                # Menampilkan grafik batang untuk kepercayaan
                plot_confidence_chart(normal_confidence, cracked_confidence)

                # Tampilkan pesan berdasarkan kepercayaan
                if result[0][predicted_class] > 0.7:
                    st.success("Model sangat yakin dengan prediksinya!")
                else:
                    st.warning("Model tidak terlalu yakin dengan prediksinya, harap cek kembali!")

            except ValueError as ve:
                st.error(f"Terjadi kesalahan pada gambar: {ve}")
            except Exception as e:
                st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
