import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load extracted features and labels
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Convert animal names into numeric labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Save the split data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Save animal class names
np.save("classes.npy", encoder.classes_)

print("Dataset splitting completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Classes:", encoder.classes_)
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)