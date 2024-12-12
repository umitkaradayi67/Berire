import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# load model
# emotion_name = ["Angry", "Disgust", "Fear",
#                 "Happy", "Sad", "Surprise", "Neutral"]
emotion_name = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
# load json and create model
json_file = open('./models/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("./models/emotion_model1.h5")
# classifier = keras.models.load_model('./models/emotion_model.h5')

# load face
try:
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # image gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        image=img_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48),
                              interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_name[maxindex]
            output = str(finalout)
        label_position = (x, y)
        cv2.putText(img, output, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Sayfa düzenini geniş yapalım
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-image: url('https://www.sdt.com.tr/uploads/images/1631718946_simulasyon-sistemleri-bilisim-teknolojileri-home.jpg');
            background-size: cover;
            background-position: center;

            video {
            background-color: rgba(0, 0, 0, 0) !important;
            opacity: 0.5; /* Burada opaklık değeri ile şeffaflık ayarlanır */
        }
    </style>
""", unsafe_allow_html=True)

# Üç kolon oluşturma
col1, col2 = st.columns([0.3, 0.7])

# Her kolona 3 kamera ekleme
with col1:
    webrtc_streamer(
        key="camera1",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # İki satır boşluk ekler
    st.markdown("<br><br>", unsafe_allow_html=True)  # İki satır boşluk ekler
    st.markdown("<br><br>", unsafe_allow_html=True)  # İki satır boşluk ekler
    webrtc_streamer(
        key="camera2",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    # Eğer session_state'de 'chat_history' yoksa, başlatalım
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = "Bot: Merhaba! Seninle sohbet etmeye hazırım.\n"

    # Kullanıcıdan mesaj almak
    user_input = st.text_input("Senin mesajın:", "")


    # Kullanıcının mesajına göre botun cevabını belirleme
    def get_bot_response(user_input):
        if "merhaba" in user_input.lower():
            return "Merhaba! Sana nasıl yardımcı olabilirim?"
        elif "nasılsın" in user_input.lower():
            return "Ben bir yapay zeka olduğum için duygularım yok ama teşekkür ederim, sen nasılsın?"
        elif "hoşça kal" in user_input.lower():
            return "Hoşça kal! Görüşmek üzere!"
        else:
            return "Üzgünüm, bunu anlamadım. Yardımcı olabilir miyim?"


    # Kullanıcı bir şey yazarsa, bot cevabı ekleyelim
    if user_input:
        bot_response = get_bot_response(user_input)
        st.session_state.chat_history += f"Sen: {user_input}\nBot: {bot_response}\n"

    # Text area içinde sohbeti göster
    st.text_area("Sohbet:", value=st.session_state.chat_history, height=300, disabled=True)
