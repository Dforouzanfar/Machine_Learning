import numpy as np
import copy, math

def zeta(x, w, b):
    z = np.dot(x, w) + b
    return z

def sigmoid(x, w, b, Z=zeta):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    zeta = Z(x, w, b)
    g = 1 / (1 + np.exp(-zeta))
    return g

def compute_cost_logistic(x, y, w, b, S=sigmoid):
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    cost = 0.0
    s = S(x, w, b)
    for i in range(m):
        cost += (-y[i] * np.log(s[i])) - ((1 - y[i]) * np.log(1 - s[i]))

    cost /= m
    return cost


def compute_gradient_logistic(X, y, w, b, S=sigmoid):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.
    s = S(X, w, b)
    err = s - y

    for i in range(m):
        for j in range(n):
            dj_dw[j] += (err[i] * X[i, j])  # scalar
        dj_db += err[i]
    dj_dw /= m  # (n,)
    dj_db /= m  # scalar

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, Z=zeta, S=sigmoid, C=compute_cost_logistic, G=compute_gradient_logistic):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    ceil = math.ceil(num_iters / 10)

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = G(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(C(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % ceil == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing