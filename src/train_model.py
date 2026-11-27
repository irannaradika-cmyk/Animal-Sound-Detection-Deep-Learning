
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
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------
# 1. PROJECT PATH
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Create required folders
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for consistent experiments
tf.keras.utils.set_random_seed(42)


# --------------------------------
# 2. LOAD FEATURES AND LABELS
# --------------------------------

X = np.load(ARTIFACTS_DIR / "X_features.npy")
y = np.load(ARTIFACTS_DIR / "y_labels.npy")

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)


# --------------------------------
# 3. ENCODE ANIMAL LABELS
# --------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

animal_classes = encoder.classes_

# Save animal class names
np.save(MODELS_DIR / "animal_classes.npy", animal_classes)

print("Animal classes:", animal_classes)


# --------------------------------
# 4. SPLIT DATASET
# --------------------------------

SPLIT_FILE = ARTIFACTS_DIR / "data_split_indices.npz"

# Use the existing split if available
if SPLIT_FILE.exists():

    print("\nLoading existing dataset split...")

    split_data = np.load(SPLIT_FILE)

    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    test_idx = split_data["test_idx"]

    # Check whether saved indices match the dataset
    all_indices = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    if (
        len(all_indices) != len(X)
        or len(np.unique(all_indices)) != len(X)
        or np.min(all_indices) < 0
        or np.max(all_indices) >= len(X)
    ):
        raise ValueError("Dataset split does not match the current dataset.")

else:

    print("\nCreating new dataset split...")

    indices = np.arange(len(X))

    # Split 20% of data for testing
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Split the remaining data into training and validation
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=42,
        stratify=y[train_val_idx]
    )

    # Save the split indices
    np.savez(
        SPLIT_FILE,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    print("Dataset split saved successfully.")


# Get training, validation and testing data
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

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

# Calculate class weights to handle imbalanced data
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# Stop training if validation loss does not improve
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Save the model with the best validation performance
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODELS_DIR / "animal_sound_cnn.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# --------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------

# Load the best saved model
model = tf.keras.models.load_model(
    MODELS_DIR / "animal_sound_cnn.keras"
)

print("\nBest 1D CNN model loaded successfully.")


# --------------------------------
# 9. TEST MODEL
# --------------------------------

y_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(y_probability, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print(
    "\nIndependent Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# --------------------------------
# 10. CLASSIFICATION REPORT
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(animal_classes)),
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
plt.show()
plt.close()


# --------------------------------
# 12. ACCURACY GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("1D CNN Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "accuracy_graph_1d.png")
plt.show()
plt.close()


# --------------------------------
# 13. LOSS GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("1D CNN Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "loss_graph_1d.png")
plt.show()
plt.close()


# --------------------------------
# 14. SAVE TEST RESULTS
# --------------------------------

np.save(
    RESULTS_DIR / "test_predictions_1d.npy",
    y_pred
)

np.save(
    RESULTS_DIR / "test_actual_labels_1d.npy",
    y_test
)

print("\nTraining and testing completed!")

print("All evaluation files saved.")
print("Project folder:", PROJECT_DIR)
