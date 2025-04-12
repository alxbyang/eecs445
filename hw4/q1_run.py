"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q1_run.py
Script for running GMM soft clustering
"""

import matplotlib.pyplot as plt
import numpy as np
import string as s

from sklearn.datasets import fetch_openml

from q1_gmm import gmm


def get_data():
    """Load penguins data from Github."""
    penguins = fetch_openml("penguins", as_frame=False)
    # get 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm' features
    X = penguins["data"][:, 1:4]
    X = X.astype(np.float32)
    # drop NA values
    X = X[~np.isnan(X).any(axis=1)]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    probs = np.array([[0.24954319], [0.39035819], [0.36009862]])
    print(f"Shape of the input data: {X.shape[0]} by {X.shape[1]}")
    return X, probs


def main():
    """Call GMM with different numbers of clusters.

    - num_K is an array containing the tested cluster sizes
    - BIC_K contains the best BIC values for each of the cluster sizes
    """
    print(
            "We'll try different numbers of clusters with GMM, using multiple runs"
            " for each to identify the 'best' results"
    )
    np.random.seed(445)
    trainX, _ = get_data()
    num_K = range(2, 9)  # List of cluster sizes
    BIC_K = np.zeros(len(num_K))
    for idx in range(len(num_K)):
        # Running
        k = num_K[idx]
        print("%d clusters..." % k)
        bestBIC = float("inf")
        # TODO: Run gmm function 10 times and select the best model for the current value of k, 
        #       storing the BIC value for that model. Use the default num_iter=10 in calling gmm()
        raise NotImplementedError

    # TODO: Part d: Make a plot to show BIC as function of clusters K
    raise NotImplementedError


if __name__ == "__main__":
    main()
