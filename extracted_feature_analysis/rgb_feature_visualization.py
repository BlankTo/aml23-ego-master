import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from load_feat import load_features_RGB


def shrink_polygon(points, shrink_factor= 0.9):
    centroid = np.mean(points, axis= 0)
    return centroid + shrink_factor * (points - centroid)

def plot_image(components, cluster_labels, central_frames, title_addon= '', save= False):

    fig, ax = plt.subplots()

    for i in range(components.shape[0]):

        central_frame = central_frames[i]
        try:
            image = plt.imread(central_frame)
            im = OffsetImage(image, zoom= 0.05)
            ab = AnnotationBbox(im, (components[i, 0], components[i, 1]), xycoords= 'data', frameon= False, zorder= 1)
            ax.add_artist(ab)
            ax.update_datalim(np.column_stack([components[i, 0], components[i, 1]]))
            ax.autoscale()
        except: continue

        #img = img.resize((img.width // 4, img.height // 4))
        #width_radius = img.width // 2
        #height_radius = img.height // 2
        #ax.imshow(img, extent=(components[i, 0] - width_radius, components[i, 0] + width_radius, components[i, 1] - height_radius, components[i, 1] + height_radius))

        #plt.scatter(components[i, 0], components[i, 1], s= 2)

    unique_labels = np.unique(cluster_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        points = np.column_stack((components[cluster_labels == label][:, 0], components[cluster_labels == label][:, 1]))
        
        if len(points) >= 3:  # ConvexHull requires at least 3 points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            shrunk_hull_points = shrink_polygon(hull_points, shrink_factor= 1.2)
            
            hull_path = np.vstack([shrunk_hull_points, shrunk_hull_points[0]])
            
            plt.plot(hull_path[:, 0], hull_path[:, 1], '-', lw= 1, zorder= 3)#, color=colors[i])

    ax.set_title(f"{title_addon} k-means shown with PCA")
    ax.set_xlabel(f'PCA Component 1')
    ax.set_ylabel(f'PCA Component 2')

    if save: plt.savefig(f"extracted_feature_analysis/images/{title_addon}", dpi=300, bbox_inches='tight')
    plt.show()

for sampling_strategy in ['dense', 'uniform']:

    for n_frames_per_clip in [5, 10, 25]:
            
            train_path = f"saved_features/D1_train_RGB_{n_frames_per_clip}_{sampling_strategy}.pkl"
            test_path = f"saved_features/D1_test_RGB_{n_frames_per_clip}_{sampling_strategy}.pkl"

            if os.path.isfile(train_path) and os.path.isfile(test_path):

                print(f'working on both {train_path} and {test_path}')

                scaler = StandardScaler()

                train_features, train_labels, train_central_frames = load_features_RGB(train_path, split= 'D1', mode= 'train', return_one_clip= True, return_central_frame= True)
                scaled_train_features = scaler.fit_transform(train_features)

                kmeans = KMeans(n_clusters= 8, random_state= 42)

                train_cluster_labels = kmeans.fit_predict(scaled_train_features)
                centroids = kmeans.cluster_centers_

                pca = PCA(n_components= 2)

                PCA_train_features = pca.fit_transform(scaled_train_features)

                test_features, test_labels, test_central_frames = load_features_RGB(test_path, split= 'D1', mode= 'test', return_one_clip= True, return_central_frame= True)
                scaled_test_features = scaler.transform(test_features)

                test_cluster_labels = kmeans.predict(scaled_test_features)

                PCA_test_features = pca.transform(scaled_test_features)

                plot_image(PCA_train_features, train_cluster_labels, train_central_frames, title_addon= f"D1_train_{n_frames_per_clip}_{sampling_strategy}", save= True)

                plot_image(PCA_test_features, test_cluster_labels, test_central_frames, title_addon= f"D1_test_{n_frames_per_clip}_{sampling_strategy}", save= True)

            elif os.path.isfile(train_path):

                print(f'working on {train_path}')

                scaler = StandardScaler()

                train_features, train_labels, train_central_frames = load_features_RGB(train_path, split= 'D1', mode= 'train', return_one_clip= True, return_central_frame= True)
                scaled_train_features = scaler.fit_transform(train_features)

                kmeans = KMeans(n_clusters= 8, random_state= 42)

                train_cluster_labels = kmeans.fit_predict(scaled_train_features)
                centroids = kmeans.cluster_centers_

                pca = PCA(n_components= 2)

                PCA_train_features = pca.fit_transform(scaled_train_features)

                plot_image(PCA_train_features, train_cluster_labels, train_central_frames, title_addon= f"D1_train_{n_frames_per_clip}_{sampling_strategy}", save= True)

            elif os.path.isfile(test_path):

                print(f'working on {test_path}')

                scaler = StandardScaler()

                test_features, test_labels, test_central_frames = load_features_RGB(test_path, split= 'D1', mode= 'test', return_one_clip= True, return_central_frame= True)
                scaled_test_features = scaler.fit_transform(test_features)

                kmeans = KMeans(n_clusters= 8, random_state= 42)

                test_cluster_labels = kmeans.fit_predict(scaled_test_features)
                centroids = kmeans.cluster_centers_

                pca = PCA(n_components= 2)

                PCA_test_features = pca.fit_transform(scaled_test_features)

                plot_image(PCA_test_features, test_cluster_labels, test_central_frames, title_addon= f"D1_test_{n_frames_per_clip}_{sampling_strategy}", save= True)

            else: print('first you have to extract the features')