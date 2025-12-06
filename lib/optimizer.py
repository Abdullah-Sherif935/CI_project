class SGD:
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad
