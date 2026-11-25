import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import sqlite3
import altair as alt
import io
from datetime import datetime
from pathlib import Path


# --------------------------------
# 1. PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Animal Sound Detection",
    page_icon="🐾",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_NAME = str(PROJECT_DIR / "detection_history.db")


# --------------------------------
# 2. DATABASE
# --------------------------------

def create_history_table():

    with sqlite3.connect(DB_NAME) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT,
                file_name TEXT,
                animal TEXT,
                confidence REAL,
                start_time REAL,
                end_time REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_audio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT,
                file_name TEXT,
                duration REAL,
                predicted_animal TEXT,
                confidence REAL,
                status TEXT
            )
        """)


def save_history(file_name, results):

    with sqlite3.connect(DB_NAME) as conn:

        for result in results:

            conn.execute("""
                INSERT INTO detection_history
                (date_time, file_name, animal, confidence,
                 start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                file_name,
                result["animal"],
                result["confidence"],
                result["start"],
                result["end"]
            ))


def save_uploaded_audio(file_name, duration, best_result):

    if best_result is not None:

        predicted_animal = best_result["animal"]
        confidence = best_result["confidence"]
        status = "Animal Detected"

    else:

        predicted_animal = "Not Detected"
        confidence = 0
        status = "No Animal Detected"

    with sqlite3.connect(DB_NAME) as conn:

        conn.execute("""
            INSERT INTO uploaded_audio_history
            (date_time, file_name, duration,
             predicted_animal, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            file_name,
            duration,
            predicted_animal,
            confidence,
            status
        ))


def get_history():

    with sqlite3.connect(DB_NAME) as conn:

        return pd.read_sql_query(
            "SELECT * FROM detection_history ORDER BY id DESC",
            conn
        )


def get_uploaded_history():

    with sqlite3.connect(DB_NAME) as conn:

        return pd.read_sql_query(
            "SELECT * FROM uploaded_audio_history ORDER BY id DESC",
            conn
        )


def clear_history():

    with sqlite3.connect(DB_NAME) as conn:

        conn.execute("DELETE FROM detection_history")
        conn.execute("DELETE FROM uploaded_audio_history")


create_history_table()


# --------------------------------
# 3. LOAD TRAINED 1D CNN MODEL
# --------------------------------

@st.cache_resource
def load_animal_model():

    model_path = PROJECT_DIR / "models" / "animal_sound_model.keras"

    # Prefer classes.npy because this is what
    # the training script saves.
    classes_path = PROJECT_DIR / "classes.npy"

    # Use the older filename only if classes.npy
    # does not exist.
    if not classes_path.exists():

        classes_path = PROJECT_DIR / "animal_classes.npy"

    if not model_path.exists():

        raise FileNotFoundError(
            "Trained model not found in the models folder."
        )

    if not classes_path.exists():

        raise FileNotFoundError(
            "Animal class file not found. "
            "Expected classes.npy or animal_classes.npy."
        )

    model = tf.keras.models.load_model(model_path)

    classes = np.load(
        classes_path,
        allow_pickle=True
    )

    if len(classes) != model.output_shape[-1]:

        raise ValueError(
            "Number of animal classes does not match "
            "the model output."
        )

    return model, classes


try:

    model, classes = load_animal_model()

except Exception as error:

    st.error(f"Model loading error: {error}")
    st.stop()


# --------------------------------
# 4. AUDIO PREPROCESSING
# --------------------------------

def preprocess_audio(audio):

    # Use a fixed 2-second audio segment.
    audio = audio[:32000]

    # Add zeros if audio is shorter than 2 seconds.
    if len(audio) < 32000:

        audio = np.pad(
            audio,
            (0, 32000 - len(audio))
        )

    # Extract Mel Spectrogram features.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )

    # Convert to decibel scale.
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# --------------------------------
# 5. CONVERT FEATURES FOR 1D CNN
# --------------------------------

def prepare_model_input(features):

    # Original shape: (64, 126)
    # Transposed shape: (126, 64)
    features = np.transpose(features, (1, 0))

    # Add batch dimension.
    # Final shape: (1, 126, 64)
    features = np.expand_dims(
        features,
        axis=0
    )

    return features.astype(np.float32)


# --------------------------------
# 6. TIME FORMATTING
# --------------------------------

def format_audio_time(seconds):

    total_seconds = max(0.0, float(seconds))

    minutes = int(total_seconds // 60)

    remaining_seconds = total_seconds % 60

    return f"{minutes:02d}:{remaining_seconds:04.1f}"


# --------------------------------
# 7. ANIMAL DETECTION
# --------------------------------

def detect_animals(audio, sr):

    window_size = 2 * sr

    step_size = int(0.5 * sr)

    results = []

    if len(audio) == 0:

        return results

    for start in range(0, len(audio), step_size):

        end = min(
            start + window_size,
            len(audio)
        )

        chunk = audio[start:end]

        if len(chunk) == 0:

            continue

        # Extract audio features.
        features = preprocess_audio(chunk)

        # Convert features for 1D CNN.
        features = prepare_model_input(features)

        # Predict animal.
        prediction = model.predict(
            features,
            verbose=0
        )[0]

        index = np.argmax(prediction)

        results.append({

            "animal": str(classes[index]),

            "confidence": float(
                prediction[index] * 100
            ),

            "start": start / sr,

            "end": end / sr

        })

        if end == len(audio):

            break

    return results


# --------------------------------
# 8. MERGE DETECTIONS
# --------------------------------

def merge_detections(results):

    filtered = [

        result for result in results

        if result["confidence"] >= 60

    ]

    merged = []

    for result in filtered:

        if (

            merged

            and merged[-1]["animal"] == result["animal"]

            and result["start"] <= merged[-1]["end"] + 0.5

        ):

            merged[-1]["end"] = max(
                merged[-1]["end"],
                result["end"]
            )

            merged[-1]["confidence"] = max(
                merged[-1]["confidence"],
                result["confidence"]
            )

        else:

            merged.append(result.copy())

    return merged


# --------------------------------
# 9. CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f6fc;
}

h1 {
    color: #253858;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #3949ab;
}

.stButton > button {
    background-color: #5146d8;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #3528a8;
    color: white;
}

.stAlert {
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e0e4f0;
}

[data-testid="stFileUploader"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px dashed #5146d8;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 10. APPLICATION HEADER
# --------------------------------

st.title("🐾 Animal Sound Classification")

st.markdown("""
### Welcome to Animal Sound Detection!

Upload an audio recording and let the AI model
identify whether it contains a dog, cat or sheep sound.

The application uses:

- Log-Mel Spectrogram feature extraction
- Deep Learning with a 1D CNN
- Animal sound classification
- Confidence score and detection history
""")

st.info(
    "Supported animals: Dog 🐶 | Cat 🐱 | Sheep 🐑"
)

st.divider()


# --------------------------------
# 11. AUDIO UPLOAD
# --------------------------------

st.subheader("🎵 Upload Animal Audio")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac"]
)

if uploaded_file is not None:

    st.audio(uploaded_file.getvalue())


predict_button = st.button(
    "🔍 Predict Animal",
    use_container_width=True
)


# --------------------------------
# 12. PREDICTION
# --------------------------------

if uploaded_file is not None and predict_button:

    audio_bytes = uploaded_file.getvalue()

    try:

        with st.spinner("Detecting animal sounds..."):

            # Load audio at 16 kHz.
            audio, sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=16000,
                mono=True
            )

            if len(audio) == 0:

                st.error("Uploaded audio is empty.")

            else:

                duration = len(audio) / sr

                # Extract features from the first 2 seconds.
                overall_features = preprocess_audio(audio)

                # Convert to 1D CNN input format.
                overall_features = prepare_model_input(
                    overall_features
                )

                # Predict the animal.
                overall_prediction = model.predict(
                    overall_features,
                    verbose=0
                )[0]

                predicted_index = np.argmax(
                    overall_prediction
                )

                predicted_animal = str(
                    classes[predicted_index]
                )

                confidence = float(
                    overall_prediction[predicted_index] * 100
                )

                # Detect animal sounds in audio windows.
                raw_results = detect_animals(
                    audio,
                    sr
                )

                results = merge_detections(
                    raw_results
                )

                best_result = {

                    "animal": predicted_animal,

                    "confidence": confidence

                }

                # Save uploaded audio history.
                save_uploaded_audio(
                    uploaded_file.name,
                    duration,
                    best_result
                )

                # Save individual detection records.
                save_history(
                    uploaded_file.name,
                    results
                )

                # Store result for display.
                st.session_state["last_prediction"] = {

                    "file_name": uploaded_file.name,

                    "animal": predicted_animal,

                    "confidence": confidence,

                    "duration": duration,

                    "probabilities": overall_prediction * 100,

                    "results": results

                }

    except Exception as error:

        st.error(
            f"Audio processing error: {error}"
        )


# --------------------------------
# 13. DISPLAY PREDICTION
# --------------------------------

if "last_prediction" in st.session_state:

    result = st.session_state["last_prediction"]

    st.divider()

    st.subheader("🐾 Prediction Result")

    st.success(
        f"Predicted Animal: {result['animal'].capitalize()}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Animal",
        result["animal"].capitalize()
    )

    col2.metric(
        "Confidence Score",
        f"{result['confidence']:.2f}%"
    )

    col3.metric(
        "Audio Duration",
        f"{result['duration']:.2f} sec"
    )

    st.caption(
        f"Audio file: {result['file_name']}"
    )

    # Confidence chart.
    st.subheader("📊 Confidence for Each Animal")

    chart_data = pd.DataFrame({

        "Animal": [

            str(animal).capitalize()

            for animal in classes

        ],

        "Confidence (%)": result["probabilities"]

    })

    chart = alt.Chart(chart_data).mark_bar(
        cornerRadiusEnd=8
    ).encode(

        x=alt.X(
            "Confidence (%):Q",
            title="Confidence Score (%)",
            scale=alt.Scale(domain=[0, 100])
        ),

        y=alt.Y(
            "Animal:N",
            title="Animal",
            sort="-x"
        ),

        color="Animal:N",

        tooltip=[
            "Animal",
            "Confidence (%)"
        ]

    )

    st.altair_chart(
        chart,
        use_container_width=True
    )

    # Detection intervals.
    st.subheader("🐾 Detected Animal Time Intervals")

    st.caption(
        "Approximate intervals estimated from overlapping "
        "2-second audio windows."
    )

    if result["results"]:

        result_df = pd.DataFrame(
            result["results"]
        )

        result_df = result_df.rename(

            columns={

                "animal": "Animal",

                "confidence": "Confidence (%)",

                "start": "Start (sec)",

                "end": "End (sec)"

            }

        )

        result_df["Animal"] = (
            result_df["Animal"].str.capitalize()
        )

        result_df["Start Time"] = (
            result_df["Start (sec)"].apply(format_audio_time)
        )

        result_df["End Time"] = (
            result_df["End (sec)"].apply(format_audio_time)
        )

        result_df["Duration (sec)"] = (

            result_df["End (sec)"]
            - result_df["Start (sec)"]

        ).round(2)

        st.dataframe(

            result_df[

                [

                    "Animal",

                    "Start Time",

                    "End Time",

                    "Duration (sec)",

                    "Confidence (%)"

                ]

            ],

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "No animal sound detected above 60% confidence."
        )


# --------------------------------
# 14. DETECTION HISTORY
# --------------------------------

st.divider()

st.header("📜 Detailed Detection History")

file_history = get_uploaded_history()

detection_history = get_history()


if file_history.empty and detection_history.empty:

    st.info("No prediction history available yet.")

else:

    # Summary information.
    total_audio_files = len(file_history)

    total_detections = len(detection_history)

    detected_files = (

        len(

            file_history[

                file_history["status"] == "Animal Detected"

            ]

        )

        if not file_history.empty

        else 0

    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Audio Files Tested",
        total_audio_files
    )

    col2.metric(
        "Files with Animal Detection",
        detected_files
    )

    col3.metric(
        "Total Detection Segments",
        total_detections
    )

    # --------------------------------
    # ALL UPLOADED AUDIO FILES
    # --------------------------------

    st.subheader("🎵 All Tested Audio Files")

    if not file_history.empty:

        file_display = file_history.copy()

        file_display["Duration (sec)"] = (
            file_display["duration"].round(2)
        )

        file_display = file_display.rename(

            columns={

                "date_time": "Date & Time",

                "file_name": "Audio Filename",

                "predicted_animal": "Predicted Animal",

                "confidence": "Confidence (%)",

                "status": "Status"

            }

        )

        st.dataframe(

            file_display[

                [

                    "Date & Time",

                    "Audio Filename",

                    "Duration (sec)",

                    "Predicted Animal",

                    "Confidence (%)",

                    "Status"

                ]

            ],

            use_container_width=True,

            hide_index=True

        )

        audio_csv = file_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "⬇️ Download Audio History",

            data=audio_csv,

            file_name="animal_audio_history.csv",

            mime="text/csv"

        )

    else:

        st.info("No uploaded audio records available.")

    st.divider()

    # --------------------------------
    # INDIVIDUAL DETECTION RECORDS
    # --------------------------------

    st.subheader("🐾 Individual Detection Records")

    if not detection_history.empty:

        detail_display = detection_history.copy()

        detail_display["Start Time"] = (
            detail_display["start_time"].apply(format_audio_time)
        )

        detail_display["End Time"] = (
            detail_display["end_time"].apply(format_audio_time)
        )

        detail_display["Duration (sec)"] = (

            detail_display["end_time"]

            - detail_display["start_time"]

        ).round(2)

        detail_display["animal"] = (
            detail_display["animal"].str.capitalize()
        )

        detail_display = detail_display.rename(

            columns={

                "date_time": "Date & Time",

                "file_name": "Audio Filename",

                "animal": "Detected Animal",

                "confidence": "Confidence (%)"

            }

        )

        st.dataframe(

            detail_display[

                [

                    "Date & Time",

                    "Audio Filename",

                    "Detected Animal",

                    "Confidence (%)",

                    "Start Time",

                    "End Time",

                    "Duration (sec)"

                ]

            ],

            use_container_width=True,

            hide_index=True

        )

        detection_csv = detail_display.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "⬇️ Download Detailed Detection History",

            data=detection_csv,

            file_name="animal_detection_details.csv",

            mime="text/csv"

        )

    else:

        st.info(
            "No animal detection segments are available."
        )

    st.divider()

    # --------------------------------
    # CLEAR HISTORY
    # --------------------------------

    st.subheader("🗑️ Manage History")

    if st.button(
        "Clear All Prediction History",
        use_container_width=True
    ):

        clear_history()

        st.session_state.pop(
            "last_prediction",
            None
        )

        st.success(
            "All prediction history has been cleared."
        )

        st.rerun()
