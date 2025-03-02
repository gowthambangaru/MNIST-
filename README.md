# MNIST-# MNIST Feedforward Neural Network

## Overview
This project implements a flexible feedforward neural network for the MNIST dataset using PyTorch. It supports multiple hidden layers, different activation functions, various optimizers, and batch sizes. The model is trained, validated, and tested using different hyperparameters.

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation
To install the required dependencies, run:
## Dataset
The MNIST dataset is automatically downloaded when running the script. It consists of handwritten digits (0-9) and is split into training, validation, and test sets.

## How to Run
1. Run `main.py` to train the model with default parameters.
2. The training process will display loss and validation accuracy per epoch.
3. The final test accuracy and confusion matrix will be generated.

## Configurable Hyperparameters
Modify `main.py` to change:
- Number of hidden layers
- Number of neurons per layer
- Learning rate
- Batch size
- Optimizer (`sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`)
- Activation function (`relu`, `sigmoid`)
- Weight initialization (`random`, `xavier`)

## Results
- Best model configuration: **[Hidden Layers: 3, Neurons: 128, Optimizer: Adam]**
- Test Accuracy: **~98%**
- Cross-entropy loss vs. Mean Squared Error loss is compared.

## Report
The report includes:
- Accuracy trends across hyperparameters
- Confusion matrix analysis
- Recommendations for MNIST

## Notes
- Ensure you have a GPU (if available) for faster training.
- All datasets are stored in `./data/`.
- The model checkpoint is saved after training.

## Contact
For any issues, reach out via email or GitHub.
