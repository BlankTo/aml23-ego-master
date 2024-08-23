import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from load_feat_2 import load_features_RGB, scale_features, get_colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def tsne_analysis(features_scaled, labels, k= 8, name_addon=''):

    label_set = set(labels)

    colors, color_map = get_colors(labels)

    if False:

        pca = PCA(n_components= 400)
        features_scaled = pca.fit_transform(features_scaled)

    if False:

        lda = LinearDiscriminantAnalysis(n_components= len(set(labels)) - 1)
        features_scaled = lda.fit_transform(features_scaled, labels)

    tsne = TSNE(n_components= 2, random_state= 42, verbose= True)
    tsne_results = tsne.fit_transform(features_scaled)

    for i in range(len(labels)):
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=colors[i], s= 2)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization after PCA-400-LDA-17')
    plt.show()


if __name__ == '__main__':

    features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    tsne_analysis(features_scaled, labels, name_addon= 'prova')