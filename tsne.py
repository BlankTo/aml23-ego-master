import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from load_feat import load_features, scale_features, get_colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def tsne_analysis(features_scaled, labels, k= 8, name_addon=''):

    label_set = set(labels)

    colors, color_map = get_colors(labels)

    pca = PCA(n_components= 400)
    components = pca.fit_transform(features_scaled)
    colors, color_map = get_colors(labels)

    lda = LinearDiscriminantAnalysis(n_components= 17)
    components = lda.fit_transform(components, labels)

    tsne = TSNE(n_components= 2, random_state= 42)
    tsne_results = tsne.fit_transform(components)

    for i in range(len(labels)):
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=colors[i], s= 2)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization after PCA-400-LDA-17')
    plt.show()


if __name__ == '__main__':

    features, labels = load_features('5_frame', remove_errors= True, ret_value= 'verb')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    tsne_analysis(features_scaled, labels, name_addon= 'prova')