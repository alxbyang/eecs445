"""
EECS 445 - Introduction to Machine Learning
Winter 2024 - Homework 3
K-Means Clustering
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.cluster import KMeans

PLT_MARKERS = ['o', 's', 'D', '^', 'v', 'p', '*', 'X'] 

def visualize_kmeans(data: npt.NDArray, n_clusters: int, init: str) -> None:
    """
    Visualize the result of k-means clustering
    """

    # TODO: 3.1b Define a scikit-learn KMeans object
    # - Set argument n_clusters (number of clusters) to n_clusters
    # - Set argument init ('random' or 'k-means++') to init
    # - Set random_state to 445
    # - Set n_init to 10
    kmeans = None

    # Fit data to obtain clusters
    kmeans.fit(data)

    # TODO: Print final value of objective function ("inertia_") and include it in your writeup for 3.1c
    
    # Plot each cluster on the same axes
    plt.figure()
    for cluster in np.arange(n_clusters):
        plt.plot(data[kmeans.labels_==cluster, 0], data[kmeans.labels_==cluster, 1], PLT_MARKERS[cluster])
    plt.title(f"K-Means Clustering Visualization - {init}")
    plt.savefig(f'kmeans_visualization_{init}.png', dpi=200, bbox_inches='tight')
    print(f"Plot saved to kmeans_visualization_{init}.png")

    # Return the kmeans model
    return kmeans

def plot_inertia() -> None:
    """
    Plot inertias over a range of number of clusters.
    """
    n_clusters = np.arange(2, 13)
    inertias = []
    for k in n_clusters:
        # Keep the random state at 443 for clearer results
        clf = KMeans(k, init=init, random_state=443).fit(data) 
        # TODO: Add the inertia for each fit classifier to the list of inertias

    plt.figure()
    plt.plot(n_clusters, inertias, linestyle="-")
    plt.scatter(n_clusters, inertias, marker="o")
    plt.xlabel("Number of Clusters ($k$)")
    plt.ylabel("Inertia")
    plt.savefig("inertias.png", bbox_inches="tight")
    print(f"Plot saved to inertias.png")

def predict_new_data(noisy_data: np.ndarray, n_clusters: int, init: str) -> None:
    """
    Predict the cluster assignments for a set of heldout data.
    """

    # TODO: Utilizing the same KMeans model (with the same parameters) from visualize_kmeans, 
    # generate cluster assignments for the heldout dataset. Then fit this model on noisy data
    # provided.
    kmeans = None

    # Plot each predicted cluster
    plt.figure()
    for cluster in range(n_clusters):
        print(f"Cluster {cluster} has {noisy_data[kmeans.labels_==cluster].shape[0]} data points")
        cluster_points = noisy_data[kmeans.labels_==cluster]  # Select points in the cluster
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], PLT_MARKERS[cluster])
    plt.title(f"K-Means New Data Clusters- {init}")
    plt.savefig(f'kmeans_new_data.png', dpi=200, bbox_inches='tight')
    print(f"Plot saved to kmeans_new_data.png")


if __name__ == '__main__':
    data = pd.read_csv("q3_data/unbalanced.csv").values

    # TODO: 3.1c Set the init method as needed for the problem
    # Note: init can be set to either "random" or "k-means++"
    n_clusters = None
    init = None

    
    print("Number of clusters:", n_clusters)
    print("Initialization method:", init)

    # TODO: 3.1b + 3.1c Implement visualize_kmeans and run the script with the appropriate init value above
    kmeans_model = visualize_kmeans(data, n_clusters, init)

    # TODO: 3.1d Implement plot_inertia to generate a plot of KMeans losses with different
    # numbers of clusters. Uncomment the line below to run the code.
    # plot_inertia()

    noisy_data = pd.read_csv("q3_data/noisy.csv").values

    # TODO: 3.1e Implement predict_new_data and uncomment the line below to generated
    # clusters for the datapoints in the noisy dataset. Input new_data_n_clusters according
    # the the problem.

    new_data_n_clusters = None
    # predict_new_data(noisy_data, new_data_n_clusters, init)


