import numpy as np


class MSELoss:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        return (2.0 / batch_size) * (y_pred - y_true)
