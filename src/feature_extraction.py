
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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

# Create required folders
os.makedirs("artifacts", exist_ok=True)
os.makedirs("results", exist_ok=True)


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

np.save("artifacts/X_features.npy", X)
np.save("artifacts/y_labels.npy", y)


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


# --------------------------------
# 6. MEL SPECTROGRAM VISUALIZATION
# --------------------------------

print("\nGenerating Mel Spectrogram graphs...")

for animal in ANIMALS:

    folder_path = os.path.join(
        DATASET_PATH,
        animal
    )

    audio_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".wav")
    ])

    if not audio_files:
        print("No audio files found for", animal)
        continue

    # Select first audio sample
    file_path = os.path.join(
        folder_path,
        audio_files[0]
    )

    # Load audio
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

    # Create graph
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma"
    )

    plt.colorbar(
        format="%+2.0f dB",
        label="Power (dB)"
    )

    plt.title(
        f"Mel Spectrogram - {animal.capitalize()} Vocal Sound"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency")

    plt.tight_layout()

    # Save graph
    output_file = os.path.join(
        "results",
        f"mel_spectrogram_{animal}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Graph saved:", output_file)


# --------------------------------
# 7. FINAL MESSAGE
# --------------------------------

print("\nAll Mel Spectrogram graphs generated successfully!")
print("Check the results folder.")
