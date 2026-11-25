import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
classes = np.load("classes.npy")

# Load trained model
model = load_model("models/animal_sound_model.keras")

# Prepare input
X_test = np.transpose(X_test, (0, 2, 1))

# Predict animal sounds
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Calculate performance
print(classification_report(
    y_test,
    y_pred,
    target_names=classes
))