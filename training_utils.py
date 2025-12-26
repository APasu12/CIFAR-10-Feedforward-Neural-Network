import numpy as np
from model_logic import relu, relu_derivative, softmax, compute_loss

# Your Forward and Backward Pass Implementation
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    z1 = X.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    a2 = relu(z2)
    z3 = a2.dot(W3) + b3
    y_hat = softmax(z3)
    cache = (X, z1, a1, z2, a2, z3, y_hat)
    return y_hat, cache

def backward_pass(cache, y_true, W1, W2, W3):
    X, z1, a1, z2, a2, z3, y_hat = cache
    m = y_true.shape[0]

    dz3 = (y_hat - y_true) / m
    dW3 = a2.T.dot(dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3.dot(W3.T)
    dz2 = da2 * relu_derivative(z2)
    dW2 = a1.T.dot(dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2.dot(W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Your Training Loop
def train(X, y, X_val, y_val, W_params, b_params, rng, epochs=20, batch_size=128, learning_rate=0.01):
    W1, W2, W3 = W_params
    b1, b2, b3 = b_params
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    n_samples = X.shape[0]

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuffled, y_shuffled = X[perm], y[perm]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_hat, cache = forward_pass(X_batch, W1, b1, W2, b2, W3, b3)
            grads = backward_pass(cache, y_batch, W1, W2, W3)

            # Update parameters
            params = [W1, b1, W2, b2, W3, b3]
            for idx, grad in enumerate(grads):
                params[idx] -= learning_rate * grad
            W1, b1, W2, b2, W3, b3 = params

        y_hat_train, _ = forward_pass(X, W1, b1, W2, b2, W3, b3)
        train_loss.append(compute_loss(y, y_hat_train))
        train_acc.append(np.mean(np.argmax(y_hat_train, axis=1) == np.argmax(y, axis=1)))

        y_hat_val, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
        val_loss.append(compute_loss(y_val, y_hat_val))
        val_acc.append(np.mean(np.argmax(y_hat_val, axis=1) == np.argmax(y_val, axis=1)))

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss[-1]:.4f}, acc: {train_acc[-1]:.4f}")

    return (train_loss, train_acc, val_loss, val_acc), (W1, b1, W2, b2, W3, b3)
