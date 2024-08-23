import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from load_feat_2 import load_features_RGB, scale_features, get_colors 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def hierarchical_clustering(features_scaled, labels, name_addon=''):

    label_set = set(labels)

    colors, color_map = get_colors(labels)

    ## hierarchical clustering

    max_distance = 100

    Z = linkage(features_scaled, method= 'ward')
    dendro = dendrogram(Z)
    clusters = fcluster(Z, max_distance, criterion= 'distance')
    #print("Clusters:", clusters)

    num_clusters = len(np.unique(clusters))
    print(f"Number of clusters: {num_clusters}")
    cluster_sizes = Counter(clusters)
    print(f"Cluster sizes: {cluster_sizes}")

    dendro_order = dendro['leaves']
    for i, sample_idx in enumerate(dendro_order):
        plt.axvline(x= i*10 + 5, ymin= 0, ymax= 0.1, color= colors[sample_idx], linestyle= '-')
    plt.axhline(y= max_distance, xmin= 0, xmax= 1000, color= 'black', linestyle= '-')

    handles = [plt.Line2D([0], [0], marker= 'o', color= 'w', markerfacecolor= color_map[label], markersize= 10, linestyle= '') for label in label_set]
    plt.legend(handles, [f'{label}' for label in label_set], title= "Class Labels", loc= 'center left', bbox_to_anchor= (0.99, 0.5))

    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

    ## hierarchical clustering after PCA

    pca = PCA(n_components= 400)
    features_scaled_pca = pca.fit_transform(features_scaled)

    max_distance = 100

    Z = linkage(features_scaled_pca, method= 'ward')  # 'ward' minimizes the variance of the clusters being merged
    dendro = dendrogram(Z)
    clusters = fcluster(Z, max_distance, criterion= 'distance')
    #print("Clusters:", clusters)

    num_clusters = len(np.unique(clusters))
    print(f"Number of clusters: {num_clusters}")
    cluster_sizes = Counter(clusters)
    print(f"Cluster sizes: {cluster_sizes}")

    dendro_order = dendro['leaves']
    for i, sample_idx in enumerate(dendro_order):
        plt.axvline(x= i*10 + 5, ymin= 0, ymax= 0.01, color= colors[sample_idx], linestyle= '-')
    plt.axhline(y= max_distance, xmin= 0, xmax= 1000, color= 'black', linestyle= '-')
    
    handles = [plt.Line2D([0], [0], marker= 'o', color= 'w', markerfacecolor= color_map[label], markersize= 10, linestyle= '') for label in label_set]
    plt.legend(handles, [f'{label}' for label in label_set], title= "Class Labels", loc= 'center left', bbox_to_anchor= (0.99, 0.5))

    plt.title('Dendrogram after PCA')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

    ## hierarchical clustering after LDA

    lda = LinearDiscriminantAnalysis(n_components= len(set(labels)) - 1)
    lda = lda.fit(features_scaled, labels)
    features_scaled_lda = lda.transform(features_scaled)

    max_distance = 30

    Z = linkage(features_scaled_lda, method= 'ward')  # 'ward' minimizes the variance of the clusters being merged
    dendro = dendrogram(Z)
    clusters = fcluster(Z, max_distance, criterion= 'distance')
    #print("Clusters:", clusters)

    num_clusters = len(np.unique(clusters))
    print(f"Number of clusters: {num_clusters}")
    cluster_sizes = Counter(clusters)
    print(f"Cluster sizes: {cluster_sizes}")

    dendro_order = dendro['leaves']
    for i, sample_idx in enumerate(dendro_order):
        plt.axvline(x= i*10 + 5, ymin= 0, ymax= 0.01, color= colors[sample_idx], linestyle= '-')
    plt.axhline(y= max_distance, xmin= 0, xmax= 1000, color= 'black', linestyle= '-')
    
    handles = [plt.Line2D([0], [0], marker= 'o', color= 'w', markerfacecolor= color_map[label], markersize= 10, linestyle= '') for label in label_set]
    plt.legend(handles, [f'{label}' for label in label_set], title= "Class Labels", loc= 'center left', bbox_to_anchor= (0.99, 0.5))

    plt.title('Dendrogram after LDA')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()


if __name__ == '__main__':

    features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    hierarchical_clustering(features_scaled, labels, name_addon= 'prova')