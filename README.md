# Linear Regression with PyTorch

This project implements a simple linear regression model using PyTorch on synthetic data. It demonstrates the full machine learning workflow: generating data, normalizing inputs and outputs, defining a custom [PyTorch](https://pytorch.org/) model, training using gradient descent, making predictions, and visualizing results with matplotlib.


## 📌 Features

- Data generation based on a linear equation: `y = 46 + 2x`
- Input and output normalization for better model performance
- Custom PyTorch model using `nn.Module`
- Training using Mean Squared Error loss and Stochastic Gradient Descent (SGD)
- Prediction for unseen input (`x = 121`) with denormalized output
- Data visualization with `matplotlib`

## 🧠 Model Architecture

A single-layer linear regression model: `y = wx + b`

Implemented using:
```python
nn.Linear(in_features=1, out_features=1)
```

## How to Run

```
git clone https://github.com/your-username/linear-regression-pytorch.git
cd linear-regression-pytorch
pip install torch matplotlib numpy
python linear_regression.py
```
