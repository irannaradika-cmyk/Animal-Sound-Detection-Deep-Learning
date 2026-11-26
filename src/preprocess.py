
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# --------------------------------
# 1. SETTINGS
# --------------------------------

RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")

ANIMALS = ["cat", "dog", "sheep"]

SAMPLE_RATE = 16000
DURATION = 2
TARGET_LENGTH = SAMPLE_RATE * DURATION


# --------------------------------
# 2. AUDIO PREPROCESSING
# --------------------------------

def preprocess_audio(file_path):

    # Load audio and resample to 16000 Hz
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    # Handle empty audio
    if len(audio) == 0:
        audio = np.zeros(TARGET_LENGTH)

    # Normalize audio
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    # Padding or trimming
    if len(audio) < TARGET_LENGTH:

        padding = TARGET_LENGTH - len(audio)

        audio = np.pad(
            audio,
            (0, padding)
        )

    else:
        audio = audio[:TARGET_LENGTH]

    return audio.astype(np.float32)


# --------------------------------
# 3. PROCESS ALL DATASET FILES
# --------------------------------

def main():

    total_files = 0

    for animal in ANIMALS:

        input_folder = RAW_DIR / animal
        output_folder = PROCESSED_DIR / animal

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        if not input_folder.exists():
            print("Folder not found:", input_folder)
            continue

        for file in sorted(input_folder.iterdir()):

            if file.suffix.lower() in [
                ".wav", ".mp3", ".ogg", ".flac"
            ]:

                processed_audio = preprocess_audio(file)

                output_file = output_folder / (
                    file.stem + ".wav"
                )

                sf.write(
                    output_file,
                    processed_audio,
                    SAMPLE_RATE
                )

                total_files += 1

        print(animal, "preprocessing completed")

    print("\nTotal processed files:", total_files)


# --------------------------------
# 4. RUN PROGRAM
# --------------------------------

if __name__ == "__main__":
    main()
