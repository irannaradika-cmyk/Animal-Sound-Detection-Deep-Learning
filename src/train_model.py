
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------
# 1. LOAD FEATURES AND LABELS
# --------------------------------

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

if len(X) != len(y):
    raise ValueError("Features and labels do not match!")

# Encode animal names
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Animal classes:", encoder.classes_)

unique, counts = np.unique(y, return_counts=True)

print("\nAnimal Dataset Distribution:")

for animal, count in zip(encoder.classes_, counts):
    print(animal, ":", count)


# --------------------------------
# 2. PREPARE 1D CNN INPUT
# --------------------------------

# Original shape: (samples, 64, 126)
# New shape: (samples, 126, 64)

X = np.transpose(X, (0, 2, 1))

print("1D CNN input shape:", X.shape)


# --------------------------------
# 3. 80:20 TRAIN / TEST SPLIT
# --------------------------------

indices = np.arange(len(y))

X_train, X_test, y_train, y_test, train_indices, test_indices = (
    train_test_split(
        X,
        y,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Check no sample index occurs in both sets
assert len(set(train_indices) & set(test_indices)) == 0


# --------------------------------
# 4. VALIDATION DATA
# --------------------------------

X_train_final, X_val, y_train_final, y_val = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
)

print("Actual model training samples:", len(X_train_final))
print("Validation samples:", len(X_val))


# --------------------------------
# 5. BUILD 1D CNN MODEL
# --------------------------------

model = tf.keras.models.Sequential([

    # Input: 126 time frames, 64 Mel features
    tf.keras.layers.Input(shape=(126, 64)),

    # First convolution layer
    tf.keras.layers.Conv1D(
        32, 3, activation="relu"
    ),

    # First pooling layer
    tf.keras.layers.MaxPooling1D(2),

    # Second convolution layer
    tf.keras.layers.Conv1D(
        64, 3, activation="relu"
    ),

    # Second pooling layer
    tf.keras.layers.MaxPooling1D(2),

    # Convert features into a single vector
    tf.keras.layers.GlobalAveragePooling1D(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.3),

    # Output: 3 animal classes
    tf.keras.layers.Dense(
        3, activation="softmax"
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

model.summary()


# --------------------------------
# 7. CALCULATE CLASS WEIGHTS
# --------------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_final),
    y=y_train_final
)

class_weights = dict(
    enumerate(class_weights)
)

print("\nClass Weights:", class_weights)


# --------------------------------
# 8. TRAIN MODEL
# --------------------------------

history = model.fit(
    X_train_final,
    y_train_final,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    verbose=1
)


# --------------------------------
# 9. FINAL INDEPENDENT TEST
# --------------------------------

print("\nEvaluating on independent test data...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nIndependent Test Accuracy:",
      round(accuracy * 100, 2), "%")

print("Test Loss:", round(loss, 4))


# --------------------------------
# 10. PREDICTIONS
# --------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# --------------------------------
# 11. CLASSIFICATION REPORT
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# --------------------------------
# 12. CONFUSION MATRIX
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_))
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted Animal")
plt.ylabel("Actual Animal")
plt.title("Animal Sound Detection - Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# --------------------------------
# 13. TRAINING ACCURACY GRAPH
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
plt.title("1D CNN Model Accuracy")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("accuracy_graph.png")
plt.show()


# --------------------------------
# 14. TRAINING LOSS GRAPH
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
plt.title("1D CNN Model Loss")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("loss_graph.png")
plt.show()


# --------------------------------
# 15. SAVE MODEL AND CLASSES
# --------------------------------

model.save("animal_sound_model.keras")

np.save(
    "animal_classes.npy",
    encoder.classes_
)


# --------------------------------
# 16. SAVE TEST RESULTS
# --------------------------------

np.save("test_predictions.npy", y_pred)
np.save("test_actual_labels.npy", y_test)

print("\n================================")
print("1D CNN MODEL TRAINING COMPLETED")
print("================================")

print("Model saved: animal_sound_model.keras")
print("Classes saved: animal_classes.npy")
print("Confusion matrix saved: confusion_matrix.png")
print("Accuracy graph saved: accuracy_graph.png")
print("Loss graph saved: loss_graph.png")
print("Test predictions saved.")

print("Independent test accuracy:",
      round(accuracy * 100, 2), "%")
