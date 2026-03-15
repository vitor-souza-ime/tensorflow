# Neural Network Training Experiment with Iris Dataset

## Overview

This repository presents a simple experiment using a **Feedforward Neural Network (FNN)** implemented in Python to classify the Iris flower dataset.

The goal of the project is educational: to demonstrate how neural networks can be trained with different numbers of epochs and how this affects the model's accuracy. The project also includes visualization tools to help understand the structure of the neural network as a directed graph.

The implementation uses TensorFlow/Keras for the neural network model, scikit-learn for data preprocessing and evaluation, and NetworkX for visualizing the neural network architecture.

---

## Dataset

The experiment uses the **Iris Dataset**, one of the most widely used datasets in machine learning.

Characteristics:

* 150 samples
* 4 input features
* 3 classes of flowers:

  * Iris setosa
  * Iris versicolor
  * Iris virginica

Features:

* sepal length
* sepal width
* petal length
* petal width

The dataset is automatically loaded using `sklearn.datasets.load_iris()`.

---

## Neural Network Architecture

The neural network used in this experiment has the following structure:

Input Layer
4 neurons (features from the dataset)

Hidden Layer 1
10 neurons with ReLU activation

Hidden Layer 2
8 neurons with ReLU activation

Output Layer
3 neurons with Softmax activation (multiclass classification)

Architecture summary:

```
Input (4)
   ↓
Dense(10, ReLU)
   ↓
Dense(8, ReLU)
   ↓
Dense(3, Softmax)
```

---

## Experiment

The network is trained multiple times using different numbers of epochs:

```
10
25
50
75
100
```

For each training run:

1. The neural network is initialized
2. The model is trained
3. Predictions are made on the test set
4. Accuracy is computed
5. Results are stored for visualization

---

## Outputs

The program produces three main outputs.

### 1. Training Results (CLI)

During execution, the script prints the accuracy obtained for each number of epochs.

Example:

```
Training with 10 epochs
Accuracy: 0.88

Training with 25 epochs
Accuracy: 0.91

Training with 50 epochs
Accuracy: 0.95
```

---

### 2. Accuracy vs Epochs Plot

A graph showing how accuracy evolves as the number of training epochs increases.

This visualization helps illustrate concepts such as:

* convergence
* model learning behavior
* potential overfitting

---

### 3. Neural Network Graph

The neural network is represented as a **directed graph** where:

* each neuron is a node
* each connection between neurons is an edge

Layers are arranged vertically and sequentially:

```
Input Layer → Hidden Layer → Hidden Layer → Output Layer
```

This visualization is generated using the NetworkX library.

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/neural-network-iris-experiment.git
cd neural-network-iris-experiment
```

Install dependencies:

```
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install networkx
```

---

## Running the Project

Run the script with:

```
python main.py
```

The program will:

1. Train the neural network multiple times
2. Display accuracy results
3. Plot training results
4. Generate a graph visualization of the neural network

---

## Project Structure

```
.
├── main.py
├── README.md
```

---

## Educational Purpose

This project is designed for educational use in courses related to:

* Artificial Intelligence
* Machine Learning
* Neural Networks
* Data Science
* Computational Intelligence

It demonstrates fundamental concepts such as:

* feedforward neural networks
* dense layers
* training epochs
* accuracy evaluation
* neural network visualization

---

## License

This project is released for educational and research purposes.
