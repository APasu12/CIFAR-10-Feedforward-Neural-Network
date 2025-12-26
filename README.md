# CIFAR-10 Feedforward Neural Network from Scratch

This project implements a multi-layer feedforward neural network to classify images from the CIFAR-10 dataset using numpy for the core logic. 

## Project Features
- **Dataset**: CIFAR-10 (subsampled to 10,000 training and 2,000 test images for runtime efficiency).
- **Architecture**: 
    - Input layer: 32x32x3 images fallted to a 3072-dimensional vector.
    - Two hidden layers: 512 and 256 neurons, respectively.
    - Activation: ReLU for hidden layers and Softmax for the output layer.
- **Algorithm**: Backpropagation and Gradient Descent implemented.

## Visualizations
This project generates several plots to demonstrate the learning process:
1. **Loss and Accuracy**: Training vs. Validation performance over 10 epochs.
2. **Weight Visualization**: Displays the first 32 weight vectors of the first hidden layer.
3. **Confusion Matrix**: Evaluation of classification performance across the 10 categories.
4. **PCA Analysis**: A 2D projection of the hidden layer activations to see how the network clusters classes.


## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`
