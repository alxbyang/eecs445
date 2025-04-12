"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q1_gmm.py
The gmm function takes in as input a data matrix X and a number of gaussians in
the mixture model

The implementation assumes that the covariance matrix is shared and is a
spherical diagonal covariance matrix
"""

from scipy.stats import multivariate_normal
import numpy as np

import matplotlib.pyplot as plt


def calc_logpdf(x, mean, cov):
    """Return log probability density."""
    x = multivariate_normal.logpdf(x, mean=mean, cov=cov)
    return x


def gmm(trainX, num_K, num_iter=10, plot=False):
    """Fit a gaussian mixture model on trainX data with num_K clusters.

    trainX is a NxD matrix containing N datapoints, each with D features
    num_K is the number of clusters or mixture components
    num_iter is the maximum number of EM iterations run over the dataset

    Description of other variables:
        - mu, which is KxD, the coordinates of the means
        - p, which is Kx1 and represents the cluster proportions
        - z, which is NxK, has at each z(n,k) the probability that the nth
          data point belongs to cluster k, specifying the cluster associated
          with each data point
        - si2 is the estimated (shared) variance of the data
        - BIC is the Bayesian Information Criterion (smaller BIC is better)
    """
    N = trainX.shape[0]
    D = trainX.shape[1]

    if num_K >= N:
        print("You are trying too many clusters")
        raise ValueError
    if plot and D != 2:
        print("Can only visualize if D = 2")
        raise ValueError

    si2 = 5  # Initialization of variance
    p = np.ones((num_K, 1)) / num_K  # Uniformly initialize cluster proportions
    mu = np.random.randn(num_K, D)  # Random initialization of clusters
    z = np.zeros(
        [N, num_K]
    )  # Matrix containing cluster membership probability for each point

    if plot:
        plt.ion()
        fig = plt.figure()
    for i in range(0, num_iter):
        """Iterate through one loop of the EM algorithm."""
        if plot:
            plt.clf()
            xVals = trainX[:, 0]
            yVals = trainX[:, 1]
            x = np.linspace(np.min(xVals), np.max(xVals), 500)
            y = np.linspace(np.min(yVals), np.max(yVals), 500)
            X, Y = np.meshgrid(x, y)
            pos = np.array([X.flatten(), Y.flatten()]).T
            plt.scatter(xVals, yVals, color="black")
            pdfs = []
            for k in range(num_K):
                rv = multivariate_normal(mu[k], si2)
                pdfs.append(rv.pdf(pos).reshape(500, 500))
            pdfs = np.array(pdfs)
            plt.contourf(X, Y, np.max(pdfs, axis=0), alpha=0.8)
            plt.pause(0.01)

        """
        E-Step
        In the first step, we find the expected log-likelihood of the data
        which is equivalent to:
        Finding cluster assignments for each point probabilistically
        In this section, you will calculate the values of z(n,k) for all n and
        k according to current values of si2, p and mu
        """
        # TODO: Implement the E-step by calculating the values of z
        raise NotImplementedError

        """
        M-step
        Compute the GMM parameters from the expressions which you have in the spec
        """

        # TODO: Estimate new value of p
        raise NotImplementedError

        # TODO: Estimate new value for mu's
        raise NotImplementedError

        # TODO: Estimate new value for sigma^2
        raise NotImplementedError

    if plot:
        plt.ioff()
        plt.savefig('visualize_clusters.png')

    # TODO: Compute the expected log-likelihood of data for the optimal parameters computed
    raise NotImplementedError
    loglikelihood = 0.0

    # TODO: Compute the BIC for the current clustering
    raise NotImplementedError
    BIC = None

    return mu, p, z, si2, float(BIC)

def gaussian(xi, mu, si2):
    D = xi.shape[0]
    return (1.0 / np.sqrt((2.0 * np.pi * si2) ** D)) * np.exp(-np.sum((xi - mu) ** 2) / (2 * si2))
