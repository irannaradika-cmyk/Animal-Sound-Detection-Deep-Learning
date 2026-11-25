
import os
import librosa
import numpy as np

DATASET_PATH = "dataset/processed"
ANIMALS = ["dog", "cat", "sheep"]

SAMPLE_RATE = 16000
N_MELS = 64
TARGET_LENGTH = SAMPLE_RATE * 2  # 2 seconds = 32000 samples

X = []
y = []
file_names = []

for animal in ANIMALS:

    folder_path = os.path.join(DATASET_PATH, animal)
    files = sorted(os.listdir(folder_path))

    for file in files:

        if file.lower().endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            # 1. Load and resample audio to 16 kHz   (Resampling sr = 16000)
            audio, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE
            )

            # 2. Normalize audio volume  (Normalization)
            audio = librosa.util.normalize(audio)

            # 3. Remove silence from beginning and end  (Trimming)
            audio, _ = librosa.effects.trim(audio)

            # 4. Make audio exactly 2 seconds long
            if len(audio) > TARGET_LENGTH:
                audio = audio[:TARGET_LENGTH]

            elif len(audio) < TARGET_LENGTH:
                audio = np.pad(                      #  (padding)
                    audio,
                    (0, TARGET_LENGTH - len(audio))
                )

            # Extract Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS,
                n_fft=1024,
                hop_length=256
            )

            # Convert to Log-Mel Spectrogram
            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            X.append(mel_db)
            y.append(animal)
            file_names.append(file_path)

    print(f"{animal} features extracted")

X = np.array(X)
y = np.array(y)
file_names = np.array(file_names)

np.save("X_features.npy", X)
np.save("y_labels.npy", y)
np.save("feature_file_names.npy", file_names)

print("\nFeature extraction completed!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("File names saved:", len(file_names))
print("Animal classes:", np.unique(y))
