import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model("animal_sound_model.keras")

classes = np.load("classes.npy", allow_pickle=True)

st.title("🐾 Animal Sound Detection")

st.write("Upload an audio file to identify the animal sound.")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    st.success("Audio uploaded successfully!")

def preprocess_audio(audio_file):

    audio, sr = librosa.load(audio_file, sr=16000, mono=True)

    audio = audio[:32000]

    if len(audio) < 32000:
        audio = np.pad(audio, (0, 32000 - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    return mel    
if uploaded_file is not None:

    features = preprocess_audio(uploaded_file)

    features = np.expand_dims(features, axis=-1)

    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)

    predicted_index = np.argmax(prediction)

    animal = classes[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    st.subheader("🐾 Prediction Result")

    st.success(f"Detected Animal: {animal.capitalize()}")

    st.write(f"Confidence: {confidence:.2f}%")