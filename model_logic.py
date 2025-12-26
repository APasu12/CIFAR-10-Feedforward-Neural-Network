import numpy as np

# Your Network Architecture and Parameter Initialization
def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size, rng):
    W1 = rng.standard_normal((input_size, hidden1_size)) * 0.01 
    b1 = np.zeros((1, hidden1_size))
    W2 = rng.standard_normal((hidden1_size, hidden2_size)) * 0.01 
    b2 = np.zeros((1, hidden2_size))
    W3 = rng.standard_normal((hidden2_size, output_size)) * 0.01 
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3

# Your activation functions and their derivatives
def relu(z): 
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z): 
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    clipped = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(clipped)) / m
