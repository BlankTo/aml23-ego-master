import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from load_feat import load_features, scale_features, get_colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def som_analysis(features_scaled, labels, name_addon=''):

    label_set = set(labels)

    pca = PCA(n_components= 400)
    features_scaled = pca.fit_transform(features_scaled)

    lda = LinearDiscriminantAnalysis(n_components= 17)
    features_scaled = lda.fit_transform(features_scaled, labels)

    som_x, som_y = 30, 30

    som = MiniSom(som_x, som_y, features_scaled.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(features_scaled)
    som.train_random(features_scaled, 10000)

    ax = plt.figure(figsize=(12, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.5)  # Distance map
    plt.colorbar(label='Distance')
    
    colors, color_map = get_colors(labels)

    # data points with color based on labels
    for i, x in enumerate(features_scaled):
        w = som.winner(x)
        plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor= colors[i], markeredgecolor= colors[i], markersize= 6, markeredgewidth= 1.5)
        
    handles = [plt.Line2D([0], [0], marker= 'o', color= 'w', markerfacecolor= color_map[label], markersize= 10, linestyle= '') for label in label_set]
    plt.legend(handles, [f'{label}' for label in label_set], title= "Class Labels", loc= 'center left', bbox_to_anchor= (1.16, 0.5))

    plt.title('SOM with Data Points Colored by Label')
    plt.show()


if __name__ == '__main__':

    features, labels = load_features('5_frame', remove_errors= True, ret_value= 'verb')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    som_analysis(features_scaled, labels, name_addon= 'prova')