# Breast Cancer Classification with MLP

A PyTorch implementation of a multilayer perceptron (MLP) for classifying breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer dataset. The model includes a 2‑unit bottleneck layer, enabling direct visualization of the learned feature space.

## Features

- **Dataset** – Wisconsin Breast Cancer dataset (569 samples, 30 features)
- **Model Architecture** – 30 → 16 → 2 (bottleneck) → 1
- **Activation** – ReLU after first hidden layer, Sigmoid at output
- **Training** – Binary Cross Entropy loss, Adam optimizer, 200 epochs
- **Visualizations** – Training loss curve & 2D bottleneck feature scatter plot
- **Performance** – Achieves 94.74% test accuracy

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Installation

```bash
pip install torch numpy matplotlib scikit-learnc
