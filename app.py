import streamlit as st
import cv2
import numpy as np
from PIL import Image
from emotion_detector import preprocess_image

st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("😊 Facial Emotion Recognition")

option = st.selectbox("Choose Input Type", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert('RGB'))
        emotion = preprocess_image(img_np)
        st.image(image, caption=f"Predicted Emotion: {emotion}", use_column_width=True)

elif option == "Use Webcam":
    run = st.button("Capture from Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    if run:
        st.warning("Press STOP in top-right to exit webcam.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image.")
                break
            emotion = preprocess_image(frame)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
