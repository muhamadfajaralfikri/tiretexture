import streamlit as st
import cv2
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

# Muat konfigurasi YOLO dan bobot pre-trained
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Fungsi untuk deteksi objek dengan YOLO
def detect_tire_yolo(image):
    # Mengubah gambar menjadi format yang dapat digunakan oleh YOLO
    blob = cv2.dnn.blobFromImage(np.array(image), 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)
    
    # Loop untuk mendeteksi objek
    height, width, _ = image.shape
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold deteksi
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                
                # Batas kotak deteksi
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

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

    # Deteksi ban menggunakan YOLO
    boxes, confidences, class_ids = detect_tire_yolo(image_np)
    
    # Pilih kotak deteksi pertama yang paling relevan (asumsi hanya ada satu ban)
    if len(boxes) > 0:
        x, y, w, h = boxes[0]
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
