import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from model_logic import initialize_parameters
from training_utils import train, forward_pass

# Load and Preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
rng = np.random.default_rng(seed=42)
train_subset, test_subset = 10000, 2000
idx_train = rng.choice(x_train.shape[0], train_subset, replace=False)
idx_test  = rng.choice(x_test.shape[0],  test_subset,  replace=False)

x_train = x_train[idx_train].reshape(train_subset, -1) / 255.0
x_test  = x_test[idx_test].reshape(test_subset, -1) / 255.0
y_train = to_categorical(y_train[idx_train], 10)
y_test  = to_categorical(y_test[idx_test], 10)

val_split = 2000
X_val, y_val = x_train[:val_split], y_train[:val_split]
X_train_sub, y_train_sub = x_train[val_split:], y_train[val_split:]

# Initialize and Train
W1, b1, W2, b2, W3, b3 = initialize_parameters(3072, 512, 256, 10, rng)
baseline, final_params = train(X_train_sub, y_train_sub, X_val, y_val, 
                               (W1, W2, W3), (b1, b2, b3), rng, 
                               epochs=10, batch_size=256, learning_rate=0.01)

# Plotting results
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(baseline[0], label='Train Loss'); plt.plot(baseline[2], label='Val Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(baseline[1], label='Train Acc'); plt.plot(baseline[3], label='Val Acc')
plt.legend()
plt.show()

# Weight Visualization
fig, axes = plt.subplots(4, 8, figsize=(12,6))
for i, ax in enumerate(axes.flatten()):
    img = final_params[0][:, i].reshape(32,32,3)
    ax.imshow((img - img.min()) / (img.max() - img.min()))
    ax.axis('off')
plt.show()
