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

    # 1. RESAMPLING:
    # Convert the audio sample rate to 16000 Hz.
    # This ensures that all audio recordings have the same sample rate.
    # mono=True converts the audio into a single channel.
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # 2. TRIMMING:
    # Remove silent portions from the beginning and end of the audio.
    # top_db=25 determines the silence threshold.
    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    # Handle empty audio after silence removal.
    if len(audio) == 0:
        audio = np.zeros(TARGET_LENGTH)

    # 3. NORMALIZATION:
    # Normalize the audio amplitude to keep the peak value
    # within the range of -1 to +1.
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    # 4. PADDING AND TRIMMING:
    # Make every audio recording exactly 2 seconds long.
    # If the audio is shorter, add zeros (padding).
    # If the audio is longer, cut it to the required length.
    if len(audio) < TARGET_LENGTH:

        padding = TARGET_LENGTH - len(audio)

        audio = np.pad(
            audio,
            (0, padding)
        )

    else:
        audio = audio[:TARGET_LENGTH]

    # Return the processed audio in float32 format.
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

                # Apply all four preprocessing operations.
                processed_audio = preprocess_audio(file)

                output_file = output_folder / (
                    file.stem + ".wav"
                )

                # Save the processed audio as a WAV file
                # with a sample rate of 16000 Hz.
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