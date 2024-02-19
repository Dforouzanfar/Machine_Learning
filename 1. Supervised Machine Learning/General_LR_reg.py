import copy, math, numpy as np

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
def compute_cost_reg(x, y, w, b, model=compute_model, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """
    m = x.shape[0]
    f_wb = model(x, w, b)
    cost = 0.
    reg_cost = 0
    error = (f_wb - y) ** 2
    for i in range(m):
        cost += error[i]  # scalar
    cost /= (2 * m)  # scalar

    if len(x.shape) == 1:
        reg_cost += (lambda_ / (2 * m)) * (w ** 2)  # scalar

    else:
        n = x.shape[1]
        for j in range(n):
            reg_cost += (w[j] ** 2)  # scalar
        reg_cost *= (lambda_ / (2 * m))  # scalar

    total_cost = cost + reg_cost  # scalar
    return total_cost

# 3
def compute_gradient_reg(x, y, w, b, model=compute_model, lambda_=1):
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
    m = x.shape[0]
    dj_db = 0
    f_wb = model(x, w, b)  # ndarray of predictions
    err = f_wb - y
    if len(x.shape) == 1:
        dj_dw = 0
        for i in range(m):
            dj_dw += err[i] * x[i]
            dj_db += err[i]
        dj_dw /= m
        dj_db /= m
        dj_dw += (lambda_/m) * w
    else:
        n = x.shape[1]
        dj_dw = np.zeros((n,))
        for i in range(m):
            for j in range(n):
                dj_dw[j] += err[i] * x[i, j]
            dj_db += err[i]
        dj_dw /= m
        dj_db /= m
        for j in range(n):
            dj_dw[j] += (lambda_/m) * w[j]  # scalar


    return dj_dw, dj_db

# 4
def gradient_descent_reg(x, y, w, b, alpha, num_iters,
                         model=compute_model,
                         cost_function=compute_cost_reg,
                         gradient_function=compute_gradient_reg):
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
    b = b
    w = copy.deepcopy(w)
    ceil = math.ceil(num_iters / 10)  # Recall: math.ceil: Round a number upward to its nearest integer:

    for i in range(num_iters + 1):
        dj_dw, dj_db = gradient_function(x, y, w, b, model)  # Calculate the gradient and update the parameters

        b -= alpha * dj_db  # Update The Parameters
        w -= alpha * dj_dw  # Update The Parameters

        cost_f = cost_function(x, y, w, b, model)  # The cost for this w and b
        J_history.append(cost_f)

        p_history.append([w, b])  # W and b

        if i % ceil == 0:  # Print cost every at intervals 1000 times (if number of  iteretaions = 100000)
            print(f"Iteration {i:4d}: Cost {J_history[-1]:0.2f}")

    return w, b, J_history, p_history  # return w and J,w history for graphing