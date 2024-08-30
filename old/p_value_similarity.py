import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from extracted_feature_analysis.old.load_feat import load_features_RGB, scale_features
from statsmodels.stats.multitest import multipletests

def p_value_similarity(features_scaled, labels):

    label_set = list(set(labels))
    len_set = len(label_set)
    labels = np.array(labels)
    print(labels.shape[0])

    heatmap_values = np.zeros((len_set, len_set))

    for i1, label_1 in enumerate(label_set):
        print(i1)
        for i2, label_2 in enumerate(label_set):
            
            features_1 = features_scaled[labels == label_1]
            features_2 = features_scaled[labels == label_2]

            p_values = []
            for feature_index in range(features_scaled.shape[1]):
                _, p_value = ttest_ind(features_1[:, feature_index], features_2[:, feature_index])
                p_values.append(p_value)

            p_values = np.array(p_values)
            corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

            similar_features = np.where(corrected_p_values > 0.95)[0]
            #print("Similar features:", similar_features)
            #print("N° Similar features:", len(similar_features))
            heatmap_values[i1, i2] = len(similar_features)

    x_labels = [f'{label}' for label in label_set]

    sns.heatmap(heatmap_values, annot= True, cmap= 'coolwarm', xticklabels= x_labels, yticklabels= x_labels)

    plt.title('p-value similarity (N° of similar features)')

    plt.show()

if __name__ == '__main__':
    
    features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train', remove_errors= True, ret_value= 'verb')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    p_value_similarity(features_scaled, labels)