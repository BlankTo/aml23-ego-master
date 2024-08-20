import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from minisom import MiniSom
from scipy.spatial.distance import cdist
from gap_statistic import gap_statistic, plot_gap_statistic
from sklearn.cluster import OPTICS

def elbow_analysys(name, name_addon=''):

    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    with open("saved_features//" + name + "_D1_test.pkl", 'rb') as f:
        saved_features = pickle.load(f)

    name = name + '_' + name_addon

    features = saved_features['features']

    clips_features = []
    for feat in features:
        clip = feat['features_RGB'][2]
        clips_features.append(clip)

    clips_features = np.array(clips_features)
    print(clips_features.shape)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clips_features)

    # OPTICS Clustering
    optics_model = OPTICS(min_samples=2)
    optics_model.fit(features_scaled)
    labels = optics_model.labels_

    # Plot results
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    plt.title('OPTICS Clustering')
    plt.show()

    exit()

    # k-mean

    ## elbow method k-mean

    inertia = []
    k_values = range(1, 20)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.plot(k_values, inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('k-mean Elbow Method')
    plt.show()

    ## silhouette method k-mean

    silhouette_scores = []
    K = range(2, 30)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features_scaled)
        silhouette_scores.append(silhouette_score(features_scaled, labels))

    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('k-mean Silhouette Analysis')
    plt.show()

    ## gap method k-mean

    def kmeans_clustering(X, k):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        return labels

    model = KMeans(n_clusters=k, random_state=42)
    gaps_kmeans = gap_statistic(features_scaled, kmeans_clustering, max_k=10, B=10)

    plot_gap_statistic(gaps_kmeans, max_k=10, text="k-means ")

    ## agglomerative

    ## elbow method agglomerative

    def calculate_distortion(X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])
        distances = cdist(X, centroids, 'euclidean')
        return np.sum(np.min(distances, axis=1))

    distortions = []
    K = range(1, 20)

    for k in K:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(features_scaled)
        distortions.append(calculate_distortion(features_scaled, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Agglomerative Clustering Elbow Method')
    plt.show()

    ## silhouette method agglomerative

    silhouette_scores = []
    K = range(2, 30)

    for k in K:
        agglo = AgglomerativeClustering(n_clusters=k)
        labels = agglo.fit_predict(features_scaled)
        silhouette_scores.append(silhouette_score(features_scaled, labels))

    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Agglomerative Clustering Silhouette Analysis')
    plt.show()

    ## gap method agglomerative

    def agglomerative_clustering(X, k):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        return labels
    
    gaps_agglomerative = gap_statistic(features_scaled, agglomerative_clustering, max_k=10, B=10)

    plot_gap_statistic(gaps_agglomerative, max_k=10, text="agglomerative ")

    # GMM

    ## elbow method GMM

    bic = []
    K = range(1, 20)
    for k in K:
        gmm = GaussianMixture(n_components=k, random_state=42).fit(features_scaled)
        bic.append(gmm.bic(features_scaled))

    plt.plot(K, bic, 'bx-')
    plt.xlabel('Number of components (k)')
    plt.ylabel('BIC')
    plt.title('GMM Elbow Method')
    plt.show()

    ## silhouette method GMM

    silhouette_scores = []
    K = range(2, 30)

    for k in K:
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(features_scaled)
        silhouette_scores.append(silhouette_score(features_scaled, labels))

    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Gaussian Mixture Models')
    plt.show()

    ## gap method GMM

    def gmm_clustering(X, k):
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X)
        return labels

    gaps_gmm = gap_statistic(features_scaled, gmm_clustering, max_k=10, B=10)

    plot_gap_statistic(gaps_gmm, max_k=10, text='GMM ')

    ##

if __name__ == '__main__':
    elbow_analysys("5_frame")