import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set direktori kerja dan path model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'cnn_model.keras')

# Muat model CNN yang sudah dilatih
model = tf.keras.models.load_model(model_path)

# Tentukan nama kelas (sesuaikan dengan pelatihan model)
class_names = ['normal', 'cracked']

# Muat model deteksi objek EfficientDet
detector_model = tf.saved_model.load("efficientdet_model/saved_model")  # Ganti dengan path model EfficientDet

# Fungsi untuk deteksi objek dengan EfficientDet
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = detector_model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # Ambil hasil deteksi
    boxes = output_dict['detection_boxes'][0].numpy()
    class_ids = output_dict['detection_classes'][0].numpy().astype(int)
    scores = output_dict['detection_scores'][0].numpy()

    return boxes, class_ids, scores

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]
    elif img_array.shape[-1] == 3:
        img_array = np.mean(img_array, axis=-1, keepdims=True)
    
    img_array = img_array.reshape((1, 224, 224, 1))
    return img_array

# Aplikasi Streamlit
st.title('Klasifikasi Tekstur Ban')

uploaded_image = st.file_uploader("Unggah gambar (lebih baik resolusi rendah)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Deteksi objek menggunakan EfficientDet
    boxes, class_ids, scores = detect_objects(image_np)

    # Filter deteksi objek untuk ban (misalnya, ID kelas ban = 1)
    tire_boxes = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and class_ids[i] == 1:  # Deteksi ban dengan skor > 0.5
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            x, y, w, h = int(xmin * image_np.shape[1]), int(ymin * image_np.shape[0]), \
                          int((xmax - xmin) * image_np.shape[1]), int((ymax - ymin) * image_np.shape[0])
            tire_boxes.append((x, y, w, h))

    # Jika ditemukan kotak deteksi ban
    if len(tire_boxes) > 0:
        x, y, w, h = tire_boxes[0]
        cropped_image = image.crop((x, y, x + w, y + h))  # Potong gambar sesuai dengan hasil deteksi

        # Tampilkan gambar yang dipotong
        st.image(cropped_image, caption="Bagian Ban yang Ditemukan", use_column_width=True)

        # Klasifikasi bagian ban
        if st.button('Klasifikasi'):
            try:
                img_array = preprocess_image(cropped_image)
                result = model.predict(img_array)

                if result.shape[-1] == 1:
                    result = np.hstack([1 - result, result])
                
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]
                
                st.success(f'Prediksi: {prediction}')
                normal_confidence = result[0][0] * 100
                cracked_confidence = result[0][1] * 100

                # Menampilkan kepercayaan
                if prediction == 'normal':
                    st.markdown(f"<h3 style='color:green;'>Kepercayaan untuk 'normal': {normal_confidence:.2f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:red;'>Kepercayaan untuk 'cracked': {cracked_confidence:.2f}%</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:red;'>Kepercayaan untuk 'normal': {normal_confidence:.2f}%</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color:green;'>Kepercayaan untuk 'cracked': {cracked_confidence:.2f}%</h3>", unsafe_allow_html=True)

                if result[0][predicted_class] > 0.7:
                    st.success("Model sangat yakin dengan prediksinya!")
                else:
                    st.warning("Model tidak terlalu yakin dengan prediksinya, harap cek kembali!")
            
            except ValueError as ve:
                st.error(f"Terjadi kesalahan pada gambar: {ve}")
            except Exception as e:
                st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
    else:
        st.warning("Tidak ditemukan ban pada gambar!")
