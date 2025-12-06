import numpy as np


def gradient_check(network, loss_fn, X, y_true, eps=1e-5):
    """
    Computes the maximum absolute difference between
    analytical gradients (backprop) and numerical gradients.
    """

    # Analytical gradients
    y_pred = network.forward(X)
    loss = loss_fn.forward(y_pred, y_true)
    grad_loss = loss_fn.backward(y_pred, y_true)
    network.backward(grad_loss)

    params = network.get_params()
    grads = network.get_grads()

    max_diff = 0.0

    # Numerical gradient
    for param, grad in zip(params, grads):
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index
            original = param[idx]

            # L(w + eps)
            param[idx] = original + eps
            loss_plus = loss_fn.forward(network.forward(X), y_true)

            # L(w - eps)
            param[idx] = original - eps
            loss_minus = loss_fn.forward(network.forward(X), y_true)

            # Restore original value
            param[idx] = original

            # Numerical derivative
            grad_num = (loss_plus - loss_minus) / (2 * eps)
            grad_ana = grad[idx]

            # Track max difference
            diff = abs(grad_num - grad_ana)
            if diff > max_diff:
                max_diff = diff

            it.iternext()

    return max_diff