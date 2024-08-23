import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from load_feat_2 import load_features_RGB, scale_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def feature_importance_score(features_scaled, labels):

    ## base features

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_scaled, labels)

    importances = clf.feature_importances_
    importances_normalized = importances / np.sum(importances)
    
    indices = np.argsort(importances_normalized)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances_normalized[indices], align='center')
    plt.yticks(range(0, len(indices), 100), [f"Feature {i}" for i in range(0, len(indices), 100)])
    plt.xlabel('Normalized Importance Score')
    plt.title('Top Feature Importances')
    plt.show()

    ## PCA 400 features

    pca = PCA(n_components= 400)
    components = pca.fit_transform(features_scaled)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(components, labels)

    importances = clf.feature_importances_
    importances_normalized = importances / np.sum(importances)
    
    indices = np.argsort(importances_normalized)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances_normalized[indices], align='center')
    plt.yticks(range(0, len(indices), 100), [f"Feature {i}" for i in range(0, len(indices), 100)])
    plt.xlabel('Normalized Importance Score')
    plt.title('Top Feature Importances - PCA 400')
    plt.show()

    ## LDA 17 features

    lda = LinearDiscriminantAnalysis(n_components= len(set(labels)) - 1)
    components = lda.fit_transform(features_scaled, labels)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(components, labels)

    importances = clf.feature_importances_
    importances_normalized = importances / np.sum(importances)
    
    indices = np.argsort(importances_normalized)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances_normalized[indices], align='center')
    plt.yticks(range(0, len(indices), 100), [f"Feature {i}" for i in range(0, len(indices), 100)])
    plt.xlabel('Normalized Importance Score')
    plt.title('Top Feature Importances - LDA 17')
    plt.show()


if __name__ == '__main__':

    features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    feature_importance_score(features_scaled, labels)