import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
classes = np.load("classes.npy")

# Add channel dimension
X_test = X_test[..., np.newaxis]

# Load trained CNN model
model = tf.keras.models.load_model("animal_sound_model.keras")

# Make predictions
predictions = model.predict(X_test)

# Convert probability predictions to class labels
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=classes,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Animal Sound Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nModel evaluation completed!")