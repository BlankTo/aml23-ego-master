import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from load_feat_2 import load_features_RGB, scale_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def find_k(features_scaled):

    max_k = 30

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis')

    ##

    pca = PCA(n_components= 400)
    components = pca.fit_transform(features_scaled)

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(components)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(components, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method PCA 400')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis PCA 400')

    ##

    pca = PCA(n_components= 2)
    components = pca.fit_transform(features_scaled)

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(components)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(components, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method PCA 2')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis PCA 2')

    ##

    lda = LinearDiscriminantAnalysis(n_components= 2)
    components = lda.fit_transform(features_scaled, labels)

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(components)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(components, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method LDA 2')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis LDA 2')

    ##

    lda = LinearDiscriminantAnalysis(n_components= len(set(labels)) - 1)
    components = lda.fit_transform(features_scaled, labels)

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(components)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(components, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method LDA 17')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis LDA 17')

    ##

    pca = PCA(n_components= 400)
    components = pca.fit_transform(features_scaled)

    lda = LinearDiscriminantAnalysis(n_components= len(set(labels)) - 1)
    components = lda.fit_transform(components, labels)

    inertia = []
    silhouette_scores = []
    k_values = range(1, max_k)
    for k in k_values:
        kmeans = KMeans(n_clusters= k, random_state= 42)
        cluster_labels = kmeans.fit_predict(components)
        inertia.append(kmeans.inertia_)
        if k > 1: silhouette_scores.append(silhouette_score(components, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw ={'hspace': 0.5})
    
    ax1.plot(k_values, inertia, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('k-mean Elbow Method PCA 400 - LDA 17')

    ax2.plot(k_values[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('k-mean Silhouette Analysis PCA 400 - LDA 17')

    plt.show()


if __name__ == '__main__':

    features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    find_k(features_scaled)