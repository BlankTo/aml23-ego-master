import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def calculate_dispersion(X, labels):
    unique_labels = np.unique(labels)
    dispersion = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            dispersion += np.sum(cdist(cluster_points, cluster_points.mean(axis=0).reshape(1, -1), 'euclidean') ** 2)
    return dispersion

def generate_reference_data(X, B):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return [np.random.uniform(mins, maxs, X.shape) for _ in range(B)]

def gap_statistic(X, clustering_function, max_k=10, B=10):
    gaps = []
    for k in range(1, max_k + 1):
        # Apply the clustering function
        labels = clustering_function(X, k)
        
        # Calculate dispersion for original data
        Wk = calculate_dispersion(X, labels)
        
        # Calculate dispersion for reference datasets
        Wk_bs = []
        reference_data = generate_reference_data(X, B)
        for ref in reference_data:
            ref_labels = clustering_function(ref, k)
            Wk_bs.append(calculate_dispersion(ref, ref_labels))
        
        Wk_bs = np.log(np.mean(Wk_bs))
        gaps.append(np.log(Wk) - Wk_bs)
    
    return gaps

def plot_gap_statistic(gaps, max_k, text=''):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), gaps, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap Statistic')
    plt.title(f'{text}Gap Statistic')
    plt.show()