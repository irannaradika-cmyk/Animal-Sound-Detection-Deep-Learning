
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# 1. Load training and testing data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# 2. Convert data for 1D CNN
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# 3. Get number of animal classes
classes = np.load("classes.npy")
num_classes = len(classes)

# 4. Build the 1D CNN model
model = Sequential([
    Conv1D(32, 3, activation="relu", padding="same",
           input_shape=(126, 64)),
    MaxPooling1D(2),

    Conv1D(64, 3, activation="relu", padding="same"),
    MaxPooling1D(2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2
)

# 7. Test the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy * 100)

# 8. Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/animal_sound_model.keras")
print("Model saved successfully!")

# 9. Save training graphs
os.makedirs("results", exist_ok=True)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_graph.png")
plt.close()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss_graph.png")
plt.close()

print("Training graphs saved successfully!")