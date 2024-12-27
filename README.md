# # MNIST Handwritten Digit Recognition

This project focuses on building, training, and evaluating a neural network model to classify handwritten digits using the MNIST dataset. The model achieves high accuracy in recognizing digits from 0 to 9, leveraging TensorFlow/Keras for implementation.

## Overview
The MNIST dataset is a benchmark dataset consisting of grayscale images of handwritten digits. Each image is 28x28 pixels and is labeled with a digit (0-9).

The goal of this project is to:
- Preprocess the MNIST dataset.
- Build and train a neural network for digit recognition.
- Evaluate the model’s performance.
- Make predictions on new data.

## Dataset
The MNIST dataset is publicly available and can be accessed via TensorFlow/Keras or directly from the official website.

- **Dataset URL**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Installation
To run this project, you need Python and the following libraries:

- TensorFlow
- NumPy
- Matplotlib

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib
```

## Implementation
### 1. Loading the Dataset
The MNIST dataset is loaded using TensorFlow:
```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

### 2. Preprocessing
- Normalize pixel values to the range [0, 1]:
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```
- Flatten the 28x28 images into 1D arrays if using a dense model.

### 3. Building the Model
Define a neural network architecture:
```python
from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 4. Training the Model
Compile and train the model:
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_split=0.1)
```

### 5. Evaluating the Model
Evaluate the model on the test set:
```python
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {accuracy}")
```

### 6. Making Predictions
Predict labels for new data:
```python
predictions = model.predict(test_images)
```

## Results
- The model achieved over **93% accuracy** on the test set.
- Visualizations of predictions and misclassified examples provide insights into model performance.

## Usage
1. Clone the repository.
2. Run the Python script to train and evaluate the model.
3. Experiment with the architecture and hyperparameters to improve accuracy.

## Acknowledgments
The MNIST dataset was created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. TensorFlow/Keras was used for the implementation.

