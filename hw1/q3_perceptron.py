"""
EECS 445 - Introduction to Machine Learning
HW1 Q3 Perceptron Algorithm with Offset
"""

import numpy as np
import numpy.typing as npt

from helper import load_data


def all_correct(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray, b: float) -> bool:
    """
    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)
        theta: normal vector of decision boundary of shape (d,)
        b: offset

    Returns:
        true if the linear classifier specified by theta and b correctly classifies all examples
    """
    return np.all((X @ theta + b) * y > 0)


def perceptron(X: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, float, npt.NDArray]:
    """
    Implements the Perception algorithm for binary linear classification.
    
    Args:
        X: array of shape (n, d) 
        y: array of shape (n,)

    Returns:
        theta: array of shape (d,)
        b: offset
        alpha: array of shape (n,). Misclassification vector, in which the i-th element is has the number of
               times the i-th point has been misclassified)
    """
    theta = np.zeros(X.shape[1])
    b = 0
    alpha = np.zeros(X.shape[0])
    
    while not all_correct(X, y, theta, b):
        for i in range(X.shape[0]):
            if (np.dot(X[i], theta) + b) * y[i] <= 0:
                theta += X[i] * y[i]
                b += y[i]
                alpha[i] += 1
    
    return theta, b, alpha


def main(fname):
    X, y = load_data(fname)
    theta, b, alpha = perceptron(X, y)

    print("Done!")
    print("============== Classifier ==============")
    print("Theta: ", theta)
    print("b: ", b)

    print("\n")
    print("============== Alpha ===================")
    print("i \t Number of Misclassifications")
    print("========================================")
    for i in range(len(alpha)):
        print(i, "\t\t", alpha[i])
    print("Total Number of Misclassifications: ", np.sum(alpha))


if __name__ == '__main__':
    main("dataset/q3.csv")
