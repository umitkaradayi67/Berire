import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.utils import img_to_array
import av

# Duygu etiketleri
emotion_name = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# Duygu tespiti modelini yükleyelim
json_file = open('./models/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("./models/emotion_model1.h5")

# Yüz tespiti için kaskad yükleyelim
try:
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
except Exception:
    st.write("Kaskad sınıflandırıcıları yüklenirken hata oluştu")

# Kamerayı başlatalım
cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # İlk kamerayı açıyoruz (0 genellikle dahili kameradır)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # av.VideoFrame'i numpy array'ine dönüştür

    # Görüntüyü gri tonlara çevirelim
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

    output = 'Yüz tespit edilmedi'  # Varsayılan çıktı
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:  # Boş görüntülerden kaçınalım
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            output = emotion_name[maxindex]

        label_position = (x, y)
        cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # İşlenmiş görüntüyü tekrar av.VideoFrame formatına dönüştürelim
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit ayarlarını yapalım
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://i.imgur.com/S1d17uV.png');

        }
    </style>
""", unsafe_allow_html=True)
# Görüntü yer tutucuları oluşturuyoruz
col1, col2 = st.columns(2)
frame_placeholder_1 = col1.empty()
frame_placeholder_2 = col2.empty()

# Kameradan gelen görüntüyü işlemeye başlayalım
if not cap.isOpened():
    st.error("Kamera açılamadı.")
else:
    while True:
        ret, frame = cap.read()  # Kameradan bir kare al

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV'den RGB formatına çevir

            # OpenCV frame'ini av.VideoFrame formatına dönüştürelim
            frame_av = av.VideoFrame.from_ndarray(frame, format="bgr24")

            # Frame'i duygu tespiti için işleyelim
            processed_frame = video_frame_callback(frame_av)

            # İşlenmiş frame'i RGB formatına çevirelim
            processed_frame_rgb = processed_frame.to_ndarray(format="bgr24")
            processed_frame_rgb = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_BGR2RGB)

            # Streamlit'teki görüntü alanlarını güncelleyelim
            frame_placeholder_1.image(frame_rgb, channels="RGB", use_column_width=True)
            frame_placeholder_2.image(processed_frame_rgb, channels="RGB", use_column_width=True)

        else:
            st.error("Görüntü alınamıyor.")

# Kamerayı serbest bırakalım
cap.release()
cv2.destroyAllWindows()