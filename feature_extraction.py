
import os
import librosa
import numpy as np

DATASET_PATH = "dataset/processed"
ANIMALS = ["dog", "cat", "sheep"]

SAMPLE_RATE = 16000
N_MELS = 64

X = []
y = []
file_names = []

for animal in ANIMALS:

    folder_path = os.path.join(DATASET_PATH, animal)

    files = sorted(os.listdir(folder_path))

    for file in files:

        if file.lower().endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            audio, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE
            )

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS,
                n_fft=1024,
                hop_length=256
            )

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
