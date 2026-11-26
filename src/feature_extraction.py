
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/processed"
ANIMALS = ["dog", "cat", "sheep"]

SAMPLE_RATE = 16000
N_MELS = 64
TARGET_LENGTH = SAMPLE_RATE * 2

file_paths = []
labels = []

# Collect original audio file paths and labels
for animal in ANIMALS:
    folder_path = os.path.join(DATASET_PATH, animal)

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(".wav"):
            file_paths.append(os.path.join(folder_path, file))
            labels.append(animal)

file_paths = np.array(file_paths)
labels = np.array(labels)

# Split original recordings: 80% training, 20% testing
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Split training recordings: 80% training, 20% validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,
    random_state=42,
    stratify=train_labels
)

# Save the original file splits
np.savez(
    "audio_splits.npz",
    train_paths=train_paths,
    train_labels=train_labels,
    val_paths=val_paths,
    val_labels=val_labels,
    test_paths=test_paths,
    test_labels=test_labels
)

print("Training files:", len(train_paths))
print("Validation files:", len(val_paths))
print("Testing files:", len(test_paths))
print("Total files:", len(file_paths))
print("Dataset split saved successfully!")

