
import os
import librosa
import numpy as np
import soundfile as sf

# ----------------------------------
# Folder paths
# ----------------------------------
RAW_FOLDER = "dataset/raw"
PROCESSED_FOLDER = "dataset/processed"

# ----------------------------------
# Settings
# ----------------------------------
SAMPLE_RATE = 16000
DURATION = 2  # seconds
SAMPLES = SAMPLE_RATE * DURATION

ANIMALS = ["dog", "cat", "sheep"]

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ----------------------------------
# Audio preprocessing function
# ----------------------------------
def preprocess_audio(file_path):

    # Load audio and convert to mono
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Remove unnecessary silence
    audio, _ = librosa.effects.trim(audio, top_db=60)

    # Normalize audio volume
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Make audio exactly 2 seconds
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    return audio


# ----------------------------------
# Process all animal audio files
# ----------------------------------
for animal in ANIMALS:

    animal_folder = os.path.join(RAW_FOLDER, animal)
    output_folder = os.path.join(PROCESSED_FOLDER, animal)

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(animal_folder):
        print(f"Folder not found: {animal_folder}")
        continue

    files = os.listdir(animal_folder)

    count = 0

    for file in files:

        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):

            input_path = os.path.join(animal_folder, file)

            try:
                audio = preprocess_audio(input_path)

                output_name = os.path.splitext(file)[0] + ".wav"
                output_path = os.path.join(output_folder, output_name)

                # Save processed audio
                sf.write(output_path, audio, SAMPLE_RATE)

                count += 1

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"{animal}: {count} files processed")

print("\nAudio preprocessing completed!")
