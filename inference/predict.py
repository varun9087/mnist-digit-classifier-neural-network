import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data.mnist_loader import load_data


# Load trained model
model = tf.keras.models.load_model("models/digit_model.h5")


# Load MNIST dataset
_, _, x_test, y_test = load_data()


# Select a test image
image = x_test[0]


# Predict digit
prediction = model.predict(image.reshape(1, 28, 28))
predicted_digit = np.argmax(prediction)


print("Predicted:", predicted_digit)
print("Actual:", y_test[0])


# Display the image
plt.imshow(image, cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")


# Save image for README
os.makedirs("images", exist_ok=True)
plt.savefig("images/prediction_example.png")

plt.show()