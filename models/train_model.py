import sys
import os

# Fix import path so Python can find the data folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.mnist_loader import load_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt


# Load dataset
x_train, y_train, x_test, y_test = load_data()


# Build neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])


# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print("\nTraining Model...\n")

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)


print("\nEvaluating Model...\n")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)


# Save trained model
os.makedirs("models", exist_ok=True)

model.save("models/digit_model.h5")

print("\nModel saved as models/digit_model.h5")


# Create images folder if not exists
os.makedirs("images", exist_ok=True)


# Plot training accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("images/training_accuracy.png")

print("\nTraining accuracy graph saved in images folder")

plt.show()