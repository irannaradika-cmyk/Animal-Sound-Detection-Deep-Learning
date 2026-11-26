
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. PROJECT PATH
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# 2. LOAD FEATURES AND LABELS
# --------------------------------

X = np.load(PROJECT_DIR / "X_features.npy")
y = np.load(PROJECT_DIR / "y_labels.npy")

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)


# --------------------------------
# 3. ENCODE ANIMAL LABELS
# --------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

animal_classes = encoder.classes_

np.save(MODELS_DIR / "animal_classes.npy", animal_classes)

print("Animal classes:", animal_classes)


# --------------------------------
# 4. SPLIT DATASET
# --------------------------------

indices = np.arange(len(X))

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.20,
    random_state=42,
    stratify=y[train_val_idx]
)

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

# Save split indices in main project folder
np.savez(
    PROJECT_DIR / "data_split_indices.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# --------------------------------
# 5. BUILD 1D CNN MODEL
# --------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2])
    ),

    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(animal_classes),
        activation="softmax"
    )
])


# --------------------------------
# 6. COMPILE MODEL
# --------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n1D CNN MODEL SUMMARY")
model.summary()


# --------------------------------
# 7. TRAIN MODEL
# --------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    verbose=1
)


# --------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------

model.save(MODELS_DIR / "animal_sound_cnn.keras")

print("\n1D CNN model saved successfully")


# --------------------------------
# 9. TEST MODEL
# --------------------------------

y_probability = model.predict(X_test)
y_pred = np.argmax(y_probability, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\nIndependent Test Accuracy:",
      round(accuracy * 100, 2), "%")


# --------------------------------
# 10. CLASSIFICATION REPORT
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=animal_classes,
        zero_division=0
    )
)


# --------------------------------
# 11. CONFUSION MATRIX
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(animal_classes))
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=animal_classes,
    yticklabels=animal_classes
)

plt.xlabel("Predicted Animal")
plt.ylabel("Actual Animal")
plt.title("1D CNN Confusion Matrix")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix_1d.png")
plt.close()


# --------------------------------
# 12. ACCURACY GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("1D CNN Training and Validation Accuracy")

plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_graph_1d.png")
plt.close()


# --------------------------------
# 13. LOSS GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("1D CNN Training and Validation Loss")

plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_graph_1d.png")
plt.close()


# --------------------------------
# 14. SAVE TEST RESULTS
# --------------------------------

np.save(RESULTS_DIR / "test_predictions_1d.npy", y_pred)
np.save(RESULTS_DIR / "test_actual_labels_1d.npy", y_test)

print("\nTraining and testing completed!")
print("All evaluation files saved.")
print("Project folder:", PROJECT_DIR)
