import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

PLT_MARKERS = ['o', 's', 'D', '^', 'v', 'p', '*', 'X'] 

def hierarchical_clustering(data: np.ndarray):
    """
    Apply Agglomerative Hierarchical Clustering and visualize clusters. This model is used
    to generate a dendrogram for 3.2a.
    """
    
    # TODO: 3.2a Define the AgglomerativeClustering class with n_clusters set to None
    # this will cause the model to generate a full linkage tree for our dendrogram. Set the
    # distance_threshold to 0.
    model = None
    model.fit(data)    

    return model  # Return the fit model

def plot_dendrogram(model: AgglomerativeClustering):
    """
    Creates a linkage matrix from an agglomerative clustering model and then plots the dendrogram.    
    """

    # This code generates the counts seen in the produced dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10, 5))

    # TODO: 3.2a Call the sklearn dendrogram function to plot the dendrogram
    # from the linkage matrix. Set truncate mode to "lastp" and set p to 5 
    # to only display the top 5 branches/clusters of the dendrogram
        

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Number of data points per branch")
    plt.ylabel("Distance")
    plt.savefig("dendrogram.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved to dendrogram.png")

def hierarchical_predict_labels(data: np.ndarray, n_clusters: int):
    """
    Apply Agglomerative Hierarchical Clustering and return fitted model and labels.
    """

    # TODO: 3.2b Implement a AgglomerativeClustering model with n_clusters set to the input n_clusters 
    # and ward linkage. Then utilize this model to fit the input data, and predict labels for cluster 
    # assignments. Note: though we could use the previously generated linkage matrix to create cluster
    # labels, this code is much simpler to implement for this assignment.

    # Hint: look into the sklearn documentation on AgglomerativeClustering and see which
    # function can be used to generate labels for data.

    model = None
    labels = None

    return model, labels

def plot_clusters(data: np.ndarray, labels: np.ndarray, n_clusters: int):
    """
    Plots the clustered data points in a 2D scatter plot.
    """
    plt.figure()
    for cluster in range(n_clusters):
        cluster_points = data[labels==cluster]  # Extract cluster points
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], PLT_MARKERS[cluster])
    plt.title("Hierarchical Clustering Visualization")
    plt.savefig("hierarchical_clusters.png", dpi=200, bbox_inches="tight")

    print(f"Cluster plot saved to hierarchical_clusters.png")

if __name__ == '__main__':

    data_csv = "q3_data/noisy.csv"

    # Load dataset
    data = pd.read_csv(data_csv).values

    print("Generating the hierarchical clustering model...")
    hc_model = hierarchical_clustering(data)

    print("Generating the dendrogram plot...")    
    plot_dendrogram(hc_model)

    # TODO: 3.2b Define the number of clusters to predict using hierarchical clustering.
    # and uncomment the lines below to generate a plot of the clusters.
    n_clusters = None

    # print("Predicting clusters and plotting points...")
    # _, labels = hierarchical_predict_labels(data, n_clusters)
    # plot_clusters(data, labels, n_clusters)
