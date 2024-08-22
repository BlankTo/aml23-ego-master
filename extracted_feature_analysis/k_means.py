import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from load_feat import load_features, scale_features, get_colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def k_means_clustering(features_scaled, labels, k= 8, name_addon=''):

    label_set = set(labels)

    colors, color_map = get_colors(labels)

    ## full features k-means

    kmeans = KMeans(n_clusters= k, random_state= 42)
    cluster_labels = kmeans.fit_predict(features_scaled)

    ## analysis

    if True:

        df = pd.DataFrame({
            'Labels': labels,
            'Cluster': cluster_labels
        })

        print(df['Cluster'].value_counts())

        for i in range(k):
            print(f"\nCluster {i}:")
            print(set(df[df['Cluster'] == i]['Labels'].values))

    ## PCA visualization of base k-means

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(len(features_scaled)):
        ax1.scatter(principal_components[i, 0], principal_components[i, 1], color= color_map[labels[i]], s= 2)#, marker= shape_map[clips_verb[i]], s= 50)
        #plt.text(principal_components[i, 0], principal_components[i, 1], f'{labels[i]}', fontsize= 5, ha= 'center', va= 'center', color= 'black')

    def shrink_polygon(points, shrink_factor= 0.9):
        centroid = np.mean(points, axis= 0)
        return centroid + shrink_factor * (points - centroid)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((principal_components[cluster_labels == label][:, 0], principal_components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            ax1.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1)#, color=colors[i])

    ax1.set_title(f'k-means shown with PCA')
    ax1.set_xlabel(f'PCA Component 1')
    ax1.set_ylabel(f'PCA Component 2')

    ax2.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis', s=2)
    ax2.set_title('PCA Visualization of Clusters')
    ax2.set_xlabel('PCA component 1')
    ax2.set_ylabel('PCA component 2')

    plt.show()

    ## LDA visualization of base k-means

    lda = LinearDiscriminantAnalysis(n_components=2)
    components = lda.fit_transform(features_scaled, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(len(features_scaled)):
        ax1.scatter(components[i, 0], components[i, 1], color= color_map[labels[i]], s= 2)#, marker= shape_map[clips_verb[i]], s= 50)
        #plt.text(components[i, 0], components[i, 1], f'{labels[i]}', fontsize= 5, ha= 'center', va= 'center', color= 'black')

    def shrink_polygon(points, shrink_factor= 0.9):
        centroid = np.mean(points, axis= 0)
        return centroid + shrink_factor * (points - centroid)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((components[cluster_labels == label][:, 0], components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            ax1.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1)#, color=colors[i])

    ax1.set_title(f'k-means shown with LDA')
    ax1.set_xlabel(f'LDA Component 1')
    ax1.set_ylabel(f'LDA Component 2')

    ax2.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='viridis', s=2)
    ax2.set_title('LDA Visualization of Clusters')
    ax2.set_xlabel('LDA component 1')
    ax2.set_ylabel('LDA component 2')

    plt.show()

    ## k-means after PCA

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    
    kmeans = KMeans(n_clusters= k, random_state= 42)
    cluster_labels = kmeans.fit_predict(principal_components)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(len(features_scaled)):
        ax1.scatter(principal_components[i, 0], principal_components[i, 1], color= color_map[labels[i]], s= 2)#, marker= shape_map[clips_verb[i]], s= 50)
        #plt.text(components[i, 0], components[i, 1], f'{labels[i]}', fontsize= 5, ha= 'center', va= 'center', color= 'black')

    def shrink_polygon(points, shrink_factor= 0.9):
        centroid = np.mean(points, axis= 0)
        return centroid + shrink_factor * (points - centroid)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((principal_components[cluster_labels == label][:, 0], principal_components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            ax1.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1)#, color=colors[i])

    ax1.set_title(f'k-means after PCA')
    ax1.set_xlabel(f'PCA Component 1')
    ax1.set_ylabel(f'PCA Component 2')

    ax2.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis', s=2)
    ax2.set_title('PCA Visualization of Clusters')
    ax2.set_xlabel('PCA component 1')
    ax2.set_ylabel('PCA component 2')

    plt.show()

    ## k-means after LDA

    lda = LinearDiscriminantAnalysis(n_components=2)
    components = lda.fit_transform(features_scaled, labels)
    
    kmeans = KMeans(n_clusters= k, random_state= 42)
    cluster_labels = kmeans.fit_predict(components)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(len(features_scaled)):
        ax1.scatter(components[i, 0], components[i, 1], color= color_map[labels[i]], s= 2)#, marker= shape_map[clips_verb[i]], s= 50)
        #plt.text(components[i, 0], components[i, 1], f'{labels[i]}', fontsize= 5, ha= 'center', va= 'center', color= 'black')

    def shrink_polygon(points, shrink_factor= 0.9):
        centroid = np.mean(points, axis= 0)
        return centroid + shrink_factor * (points - centroid)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((components[cluster_labels == label][:, 0], components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            ax1.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1)#, color=colors[i])

    ax1.set_title(f'k-means after LDA')
    ax1.set_xlabel(f'LDA Component 1')
    ax1.set_ylabel(f'LDA Component 2')

    ax2.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='viridis', s=2)
    ax2.set_title('LDA Visualization of Clusters')
    ax2.set_xlabel('LDA component 1')
    ax2.set_ylabel('LDA component 2')

    plt.show()

    ## k-means after PCA-LDA

    pca = PCA(n_components=400)
    principal_components = pca.fit_transform(features_scaled)

    lda = LinearDiscriminantAnalysis(n_components=2)
    components = lda.fit_transform(principal_components, labels)
    
    kmeans = KMeans(n_clusters= k*2, random_state= 42)
    cluster_labels = kmeans.fit_predict(components)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(len(features_scaled)):
        ax1.scatter(components[i, 0], components[i, 1], color= color_map[labels[i]], s= 2)#, marker= shape_map[clips_verb[i]], s= 50)
        #plt.text(components[i, 0], components[i, 1], f'{labels[i]}', fontsize= 5, ha= 'center', va= 'center', color= 'black')

    def shrink_polygon(points, shrink_factor= 0.9):
        centroid = np.mean(points, axis= 0)
        return centroid + shrink_factor * (points - centroid)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((components[cluster_labels == label][:, 0], components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            ax1.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1)#, color=colors[i])

    ax1.set_title(f'k-means after PCA-LDA')
    ax1.set_xlabel(f'PCA-400_LDA Component 1')
    ax1.set_ylabel(f'PCA-400_LDA Component 2')

    ax2.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='viridis', s=2)
    ax2.set_title('PCA-400_LDA Visualization of Clusters')
    ax2.set_xlabel('PCA-400_LDA component 1')
    ax2.set_ylabel('PCA-400_LDA component 2')

    plt.show()


if __name__ == '__main__':

    features, labels = load_features('5_frame', mode= 'train', remove_errors= True, ret_value= 'verb')

    features_scaled = scale_features(features, method= 'standard', ret_scaler= False)

    k_means_clustering(features_scaled, labels, name_addon= 'prova')