import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# 1. LOAD SAVED DATA
# --------------------------------

X = np.load(PROJECT_DIR / "X_features.npy")
y_labels = np.load(PROJECT_DIR / "y_labels.npy")
animal_classes = np.load(MODELS_DIR / "animal_classes.npy")

print("Feature shape:", X.shape)


# --------------------------------
# 2. LOAD SAVED TEST SPLIT
# --------------------------------

split_data = np.load(PROJECT_DIR / "data_split_indices.npz")

test_idx = split_data["test_idx"]

X_test = X[test_idx]

class_to_number = {
    animal: index
    for index, animal in enumerate(animal_classes)
}

y_test = np.array([
    class_to_number[label]
    for label in y_labels[test_idx]
])


# --------------------------------
# 3. LOAD TRAINED 1D CNN MODEL
# --------------------------------

model = tf.keras.models.load_model(
    MODELS_DIR / "animal_sound_cnn.keras"
)


# --------------------------------
# 4. PREDICT TEST SAMPLES
# --------------------------------

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# --------------------------------
# 5. CALCULATE ACCURACY
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

correct = np.sum(y_test == y_pred)
incorrect = np.sum(y_test != y_pred)

print("\n1D CNN MODEL EVALUATION")
print("-----------------------")

print("Total test samples:", len(y_test))
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)

print(
    "Test accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# --------------------------------
# 6. CLASSIFICATION REPORT
# --------------------------------

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(animal_classes)),
    target_names=animal_classes,
    zero_division=0
))


# --------------------------------
# 7. CONFUSION MATRIX
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(animal_classes))
)

print("CONFUSION MATRIX")
print(cm)


# --------------------------------
# 8. PLOT CONFUSION MATRIX
# --------------------------------

plt.figure(figsize=(7, 5))

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

plt.savefig(
    RESULTS_DIR / "confusion_matrix_1d.png",
    dpi=300
)

plt.close()


# --------------------------------
# 9. SAVE PREDICTIONS
# --------------------------------

np.save(RESULTS_DIR / "test_predictions_1d.npy", y_pred)
np.save(RESULTS_DIR / "test_actual_labels_1d.npy", y_test)


print("\nConfusion matrix saved successfully!")
print("File:", RESULTS_DIR / "confusion_matrix_1d.png")

print("\nEvaluation completed successfully!")