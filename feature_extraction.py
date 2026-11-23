import os
import librosa
import numpy as np

# Folder containing processed audio
DATASET_PATH = "dataset/processed"

# Animal classes
ANIMALS = ["dog", "cat", "sheep"]

# Mel Spectrogram settings
SAMPLE_RATE = 16000
N_MELS = 64
DURATION = 2

# Store features and labels
X = []
y = []

# Process each animal folder
for animal in ANIMALS:

    folder_path = os.path.join(DATASET_PATH, animal)

    files = os.listdir(folder_path)

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            # Load audio
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Extract Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS,
                n_fft=1024,
                hop_length=256
            )

            # Convert to decibel scale
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Store feature
            X.append(mel_db)

            # Store label
            y.append(animal)

    print(f"{animal} features extracted")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save features and labels
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nFeature extraction completed!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)