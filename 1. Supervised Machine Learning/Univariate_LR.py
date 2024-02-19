import math, numpy as np

# 1
def compute_model(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    f_wb = np.dot(x, w) + b
    return f_wb

# 2
def compute_cost(x, y, w, b, model=compute_model):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]  # number of rows
    cost = 0
    f_wb = model(x, w, b)  # ndarray of predictions
    error = (f_wb - y) ** 2

    for i in range(m):
        cost += error[i]
    total_cost = cost / (2 * m)

    return total_cost

# 3
def compute_gradient(x, y, w, b, model=compute_model):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    m = x.shape[0]  # Number of training examples
    dj_dw, dj_db = 0, 0  # Gradients
    f_wb = model(x, w, b)  # ndarray of predictions
    err = f_wb - y

    for i in range(m):
        dj_dw += err[i] * x[i]
        dj_db += err[i]
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

# 4
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, model=compute_model, cost_function=compute_cost,
                     gradient_function=compute_gradient):
    """
    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    J_history = []  # An array to store cost J at each iteration primarily for graphing later
    p_history = []  # An array to store w and b at each iteration primarily for graphing later
    b = b_in
    w = w_in
    ceil = math.ceil(num_iters / 10)  # Recall: math.ceil: Round a number upward to its nearest integer:

    for i in range(num_iters + 1):
        dj_dw, dj_db = gradient_function(x, y, w, b, model)  # Calculate the gradient and update the parameters

        b -= alpha * dj_db  # Update The Parameters
        w -= alpha * dj_dw  # Update The Parameters

        cost_f = cost_function(x, y, w, b, model)  # The cost for this w and b
        J_history.append(cost_f)

        p_history.append([w, b])  # W and b

        # Recall: 22 % 3 = 1, 22 // 3 = 7, 22 / 3 = 7.33

        if i % ceil == 0:  # Print cost every at intervals 1000 times or as many iterations if < 10
            print(f"Iteration {i:6}: Cost {J_history[-1]:0.2e} ",  # iteration i has cost j
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",  # gradients in iteration i
                  f"w: {w: 0.3e}, b:{b: 0.5e}")  # w and b in iteration i

    return w, b, J_history, p_history  # return w and J,w history for graphing