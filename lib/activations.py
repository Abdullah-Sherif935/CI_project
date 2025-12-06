import numpy as np
from .layers import Layer


class Tanh(Layer):
    """
    Tanh activation: y = tanh(x)
    """

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        """
        inputs: (batch_size, dim)
        returns: (batch_size, dim)
        """
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
        """
        grad_output: dL/dy

        dy/dx = 1 - tanh(x)^2
        so: dL/dx = dL/dy * (1 - y^2)
        """
        return grad_output * (1.0 - self.output ** 2)


class Sigmoid(Layer):
    """
    Sigmoid activation: y = 1 / (1 + exp(-x))
    """

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        """
        dy/dx = y * (1 - y)
        so: dL/dx = dL/dy * y * (1 - y)
        """
        return grad_output * self.output * (1.0 - self.output)