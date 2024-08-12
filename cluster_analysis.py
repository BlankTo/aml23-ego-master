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
import torch

def analyze_clusters(name, k_range=[32]):

    #import threadpoolctl
    #print(threadpoolctl.threadpool_info())

    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    with open("saved_features//" + name + "_D1_test.pkl", 'rb') as f:
        saved_features = pickle.load(f)

    with open("train_val//D1_test.pkl", 'rb') as f:
        train_val = pickle.load(f)

    labels = {}
    for i in range(len(train_val)):

        sample = train_val.iloc[i]
        labels[sample['uid']] = {'narration': sample['narration'], 'verb': sample['verb'], 'verb_class': sample['verb_class']}

    features = saved_features['features']

    clips_features = []
    clips_label = []
    clips_narration = []
    clips_verb = []
    clips_obj = []
    for feat in features:
        print('---------------------------------')
        print(f"video_name: {feat['video_name']}")
        print(f"uid: {feat['uid']}")
        label = labels[feat['uid']]
        print(f"narration: {label['narration']}")
        print(f"verb: {label['verb']}")
        print(f"verb_class: {label['verb_class']}")
        print(f"n_clip: {len(feat['features_RGB'])}")
        print(f"n_something_per_clip: {len(feat['features_RGB'][0])}")

        for clip in feat['features_RGB']:
            clips_features.append(clip)
            clips_label.append(label)
            clips_narration.append(label['narration'])
            clips_verb.append(label['verb'])
            clips_obj.append(label['narration'].split(' ')[-1])

    for feat, label, verb, obj in zip(clips_features, clips_label, clips_verb, clips_obj):

        print('------------------------------')
        print(label)
        print(verb)
        print(obj)
        print(len(feat))

    clips_features = np.array(clips_features)
    print(clips_features.shape)
    print(len(clips_narration))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clips_features)

    ## elbow method

    #inertia = []
    #k_values = range(1, 100)
    #for k in k_values:
    #    kmeans = KMeans(n_clusters=k, random_state=42)
    #    kmeans.fit(features_scaled)
    #    inertia.append(kmeans.inertia_)
    #
    #plt.plot(k_values, inertia, 'bx-')
    #plt.xlabel('Number of clusters (k)')
    #plt.ylabel('Inertia')
    #plt.title('Elbow Method For Optimal k')
    #plt.show()

    ## silhouette score

    #silhouette_scores = []
    #for k in range(2, 100):
    #    kmeans = KMeans(n_clusters=k, random_state=42)
    #    labels = kmeans.fit_predict(features_scaled)
    #    score = silhouette_score(features_scaled, labels)
    #    silhouette_scores.append(score)
    #
    #plt.plot(range(2, 100), silhouette_scores, 'bx-')
    #plt.xlabel('Number of clusters (k)')
    #plt.ylabel('Silhouette Score')
    #plt.title('Silhouette Method For Optimal k')
    #plt.show()

    ## clustering

    for k in k_range: # 32 from partial elbow and silhouette analysis

        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        ## analysis

        if False:

            df = pd.DataFrame({
                'Narration': clips_narration,
                'Cluster': cluster_labels
            })

            print(df['Cluster'].value_counts())

            for i in range(k):
                print(f"\nCluster {i}:")
                print(set(df[df['Cluster'] == i]['Narration'].values))

        ## PCA

        if False:

            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(features_scaled)

            plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('PCA Visualization of Clusters')
            plt.show()

        ## LDA

        if False:

            lda = LinearDiscriminantAnalysis(n_components=2)
            lda_components = lda.fit_transform(features_scaled, cluster_labels)

            colors = sns.color_palette("hsv", k)

            plt.figure(figsize=(10, 6))

            for cluster in range(k):
                plt.scatter(
                    lda_components[cluster_labels == cluster, 0],
                    lda_components[cluster_labels == cluster, 1],
                    color=colors[cluster], label=f'Cluster {cluster}', alpha=0.6
                )

            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2')
            plt.title('LDA Scatter Plot of Clusters')
            plt.legend()
            plt.show()


        ## t-SNE

        if False:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(features_scaled)

            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE Visualization of Clusters')
            plt.show()

        ## agglomerative clustering

        if False:
            agg_clustering = AgglomerativeClustering(n_clusters=5)
            cluster_labels = agg_clustering.fit_predict(features_scaled)

            linked = linkage(features_scaled, method='ward')
            plt.figure(figsize=(10, 7))
            dendrogram(linked)
            plt.title('Dendrogram for Hierarchical Clustering')
            plt.show()

        ## DBSCAN

        if False:

            dbscan = DBSCAN(eps=0.5, min_samples=1)
            cluster_labels = dbscan.fit_predict(features_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('DBSCAN Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()


        ## GMM

        if False:

            gmm = GaussianMixture(n_components=32, random_state=42)
            cluster_labels = gmm.fit_predict(features_scaled)

            lda = LinearDiscriminantAnalysis(n_components=2)
            lda_components = lda.fit_transform(features_scaled, cluster_labels)

            plt.figure(figsize=(10, 6))
            plt.scatter(lda_components[:, 0], lda_components[:, 1], c=cluster_labels, cmap='viridis', alpha= 0.7)
            plt.title('Gaussian Mixture Model Clustering')
            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2')
            plt.show()

        ## spectral clustering

        if False:

            spectral = SpectralClustering(n_clusters=32, affinity='nearest_neighbors', random_state=42)
            cluster_labels = spectral.fit_predict(features_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha= 0.7)
            plt.title('Spectral Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

        ## mean shift clustering 

        if False:

            mean_shift = MeanShift()
            cluster_labels = mean_shift.fit_predict(features_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('Mean Shift Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

        ## affinity propagation

        if False:

            affinity_prop = AffinityPropagation(random_state=42)
            cluster_labels = affinity_prop.fit_predict(features_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.title('Affinity Propagation Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

        ## SOM

        if False:

            som = MiniSom(x=50, y=50, input_len=features_scaled.shape[1], sigma=1.0, learning_rate=0.5)
            som.random_weights_init(features_scaled)
            som.train_random(features_scaled, num_iteration=100)

            # Map data points to their BMU (Best Matching Unit) on the SOM grid
            bmu_indices = np.array([som.winner(x) for x in features_scaled])

            # Assign unique color to each BMU
            unique_bmu_indices = np.unique(bmu_indices, axis=0)
            colors = plt.cm.get_cmap('viridis', len(unique_bmu_indices))

            # Plotting the data points with their corresponding BMU colors
            plt.figure(figsize=(10, 6))

            for i, bmu in enumerate(unique_bmu_indices):
                mask = np.all(bmu_indices == bmu, axis=1)
                plt.scatter(bmu_indices[mask, 0], bmu_indices[mask, 1], c=[colors(i)], label=f'Cluster {i}')

            plt.title('Self-Organizing Map Clustering')
            plt.xlabel('SOM X')
            plt.ylabel('SOM Y')
            plt.legend()
            plt.show()

        ##

        clustering_methods = {
            'KMeans': KMeans(n_clusters=k, random_state=42),
            'Agglomerative': AgglomerativeClustering(n_clusters=5),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'GaussianMixture': GaussianMixture(n_components=5, random_state=42),
            'SpectralClustering': SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42),
            'MeanShift': MeanShift(),
            'AffinityPropagation': AffinityPropagation(random_state=42),
        }

        visualization_methods = {
            'PCA': PCA(n_components=2),
            't-SNE': TSNE(n_components=2, random_state=42),
            'LDA': LinearDiscriminantAnalysis(n_components=2)
        }

        save_dir = f"clustering_results//{name}_{k}_clusters"
        os.makedirs(save_dir, exist_ok=True)

        for method_name, method in clustering_methods.items():
            
            cluster_labels = method.fit_predict(features_scaled)
            
            for vis_name, vis_method in visualization_methods.items():

                if vis_name == 'LDA':
                    if len(set(cluster_labels)) == 1: continue
                    vis_method = vis_method.fit(features_scaled, cluster_labels)
                    components = vis_method.transform(features_scaled)
                else:
                    components = vis_method.fit_transform(features_scaled)
                
                plt.figure(figsize=(10, 6))
                
                plt.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='viridis')
                plt.title(f'{method_name} with {vis_name}')
                
                plt.xlabel(f'{vis_name} Component 1')
                plt.ylabel(f'{vis_name} Component 2')
                #plt.legend()
                plt.savefig(os.path.join(save_dir, f'{name}_{method_name}_{vis_name}.png'))
                plt.close()