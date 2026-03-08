Handwritten Digit Classifier using Neural Networks
A machine learning project that builds a neural network to classify handwritten digits (0–9) using the MNIST dataset.
The model is implemented using TensorFlow/Keras and demonstrates multiclass classification with a Softmax output layer.

🚀 Project Overview
This project trains a neural network to recognize handwritten digits.
The MNIST dataset contains 70,000 grayscale images of handwritten numbers.

The model learns patterns from the training data and predicts the correct digit for unseen images.

Example prediction:

Predicted: 7
Actual: 7

🧠 Machine Learning Concepts Used
Neural Networks

ReLU Activation Function

Softmax Output Layer

Multiclass Classification

Model Training & Evaluation

Data Normalization

Model Saving and Loading

📊 Dataset
This project uses the MNIST dataset, which contains:

Dataset	Images
Training	60,000
Testing	10,000
Each image is:

28 × 28 pixels

Grayscale

Represents digits from 0 to 9

Dataset source:
https://keras.io/api/datasets/mnist/

🏗️ Project Structure
digit-classifier-neural-network
│
├── data
│ └── mnist_loader.py
│
├── models
│ ├── train_model.py
│ └── digit_model.h5
│
├── inference
│ └── predict.py
│
├── images
│ ├── sample_digit.png
│ ├── training_accuracy.png
│ └── prediction_example.png
│
├── notebooks
│ └── exploration.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md

⚙️ Installation
Clone the repository

git clone https://github.com/varun9087/mnist-digit-classifier-neural-network.git

Navigate to the project directory

cd mnist-digit-classifier-neural-network

Install dependencies

pip install -r requirements.txt

▶️ Training the Model
Run the training script:

python models/train_model.py

This will:

Download the MNIST dataset

Train the neural network

Save the trained model

Generate a training accuracy plot

Output:

models/digit_model.h5
images/training_accuracy.png

🔎 Running Predictions
To test the trained model:

python inference/predict.py

This script will:

Load the trained model

Predict a digit from the test dataset

Display the image and prediction

Save the prediction result image

Output:

images/prediction_example.png

📈 Model Architecture
Input Layer: 28×28 image
Flatten Layer: converts image to 784 features

Hidden Layer 1
Dense (128 neurons) + ReLU activation

Hidden Layer 2
Dense (64 neurons) + ReLU activation

Output Layer
Dense (10 neurons) + Softmax activation

📊 Results
Typical model accuracy:

Test Accuracy: ~97%

Example output image:


🛠️ Technologies Used
Python

TensorFlow

Keras

NumPy

Matplotlib

Jupyter Notebook

📌 Future Improvements
Possible enhancements for this project:

Add a confusion matrix

Add model evaluation metrics (precision, recall, F1-score)

Build a web interface using Flask or Streamlit

Deploy the model as an API

Implement a CNN for higher accuracy