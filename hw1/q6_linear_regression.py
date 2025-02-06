"""
EECS 445 - Introduction to Maching Learning
HW1 Q5 Linear Regression Optimization Methods)
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import time

import numpy as np
import numpy.typing as npt

from helper import load_data


def calculate_squared_loss(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray) -> float:
    """
    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)
        theta: array of shape (d,). Specifies an (d-1)^th degree polynomial
        
    Returns:
        squared loss for the given data and parameters
    """
    
    return (1/(2 * np.shape(X)[0])) * np.sum((y - X @ theta) ** 2)


def ls_gradient_descent(X: npt.NDArray, y: npt.NDArray, learning_rate: float) -> npt.NDArray:
    """
    The Gradient Descent (GD) algorithm for least squares regression.
    
    Please use the following stopping criteria together: number of iterations >= 1e6
    or |new_loss − prev_loss| <= 1e−10.

    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)
        learning_rate: the learning rate for the algorithm
    
    Returns:
        theta: array of shape (d,)
    """
    n, d = X.shape
    theta = np.zeros(d)

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)


    while n_iter < max_iter and abs(new_loss - prev_loss) > eps:  # TODO: Implement the correct stopping criteria
        n_iter += 1
        grad = []

        for xt, yt in zip(X, y):
            grad.append(-1 * (yt-(xt@theta)) * xt)  # TODO: Append the gradient of the loss function evaluated at each point


        theta -= step * np.mean(np.array(grad), axis=0)  # Update theta using the average gradient

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta


def ls_stochastic_gradient_descent(X: npt.NDArray, y: npt.NDArray, learning_rate: float) -> npt.NDArray:
    """
    The Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    
    Please do not shuffle your data points.
    
    Please use the following stopping criteria together: number of iterations >= 1e6
    or |new_loss − prev_loss| <= 1e−10.
    
    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)
        learning_rate: the learning rate for the algorithm
    
    Returns:
        theta: array of shape (d,)
    """
    n, d = X.shape
    theta = np.zeros(d)
    adaptive = (learning_rate == 'adaptive')

    eps = 1e-10
    max_iter = 1e6
    n_iter = 0
    epochs = 0

    step = learning_rate
    prev_loss = np.inf
    new_loss = calculate_squared_loss(X, y, theta)


    while n_iter < max_iter and abs(new_loss - prev_loss) > eps:  # TODO: Implement the correct stopping criteria
        epochs += 1
        if adaptive:
            step = 1 / (1 + epochs)  # TODO: [5d] Implement adaptive learning rate update step 

        for xt, yt in zip(X, y):
            theta -= step * -1 * (yt-(xt@theta)) * xt  # TODO: Implement the update step
            n_iter += 1

        prev_loss = new_loss
        new_loss = calculate_squared_loss(X, y, theta)

    print("Learning rate:", learning_rate, "\t\t\tNum iterations:", n_iter)
    return theta


def ls_closed_form_optimization(X: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """
    Implement the closed form solution for least squares regression.

    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)

    Returns:
        theta: array of shape (d,)
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def main(fname_train):
    # TODO: This function should contain all the code you implement to complete question 5.

    X_train, y_train = load_data(fname_train)
    
    # Appending a column of constant ones to the X_train matrix to make X_train the same dimensions as theta.
    # The term multiplied by theta_0 is x^0 = 1 (theta_0 is a constant), which is why the column contains only ones.
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    learning_rate = 1e-1
    
    
    print("GRADIENT DESCENT")
    start_time = time.process_time()
    theta_gd = ls_gradient_descent(X_train, y_train, learning_rate)
    end_time = time.process_time()
    print("Theta:", theta_gd, "\tTime: {:.6f} seconds".format(end_time - start_time), "\n")
    
    print("STOCHASTIC GRADIENT DESCENT")
    start_time = time.process_time()
    theta_sgd = ls_stochastic_gradient_descent(X_train, y_train, 'adaptive')
    end_time = time.process_time()
    print("Theta:", theta_sgd, "\tTime: {:.6f} seconds".format(end_time - start_time), "\n")
    
    # print("CLOSED FORM")
    # start_time = time.process_time()
    # theta_closed = ls_closed_form_optimization(X_train, y_train)
    # end_time = time.process_time()
    # print("Theta:", theta_closed, "\tTime: {:.6f} seconds".format(end_time - start_time), "\n")

    print("Done!")


if __name__ == '__main__':
    main("dataset/q6.csv")
