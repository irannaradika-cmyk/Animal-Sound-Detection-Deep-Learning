
import os
import librosa
import numpy as np

# --------------------------------
# 1. SETTINGS
# --------------------------------

DATASET_PATH = "dataset/processed"

ANIMALS = ["cat", "dog", "sheep"]

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

X = []
y = []


# --------------------------------
# 2. FEATURE EXTRACTION
# --------------------------------

for animal in ANIMALS:

    folder_path = os.path.join(
        DATASET_PATH,
        animal
    )

    for file in sorted(os.listdir(folder_path)):

        if file.lower().endswith(".wav"):

            file_path = os.path.join(
                folder_path,
                file
            )

            # Load processed audio
            audio, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE
            )

            # Generate Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )

            # Convert power to decibels
            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # Transpose for 1D CNN
            # Shape: (time frames, 64 features)
            mel_sequence = mel_db.T

            X.append(mel_sequence)
            y.append(animal)

    print(animal, "features extracted")


# --------------------------------
# 3. CONVERT TO NUMPY ARRAYS
# --------------------------------

X = np.array(X, dtype=np.float32)
y = np.array(y)


# --------------------------------
# 4. SAVE FEATURES AND LABELS
# --------------------------------

np.save("X_features.npy", X)
np.save("y_labels.npy", y)


# --------------------------------
# 5. DISPLAY RESULTS
# --------------------------------

print("\nFeature extraction completed")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("\nAnimal distribution:")

for animal in ANIMALS:
    print(animal, ":", np.sum(y == animal))

print("\nSaved X_features.npy and y_labels.npy")
