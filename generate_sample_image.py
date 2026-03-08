import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(_, _), (x_test, y_test) = mnist.load_data()

plt.imshow(x_test[0], cmap="gray")
plt.title(f"Digit: {y_test[0]}")
plt.axis("off")

plt.savefig("images/sample_digit.png")