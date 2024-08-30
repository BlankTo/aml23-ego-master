import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from minisom import MiniSom
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from collections import Counter

def load_features_RGB(name, split= 'D1', mode= 'train', cleanse= False, remove_errors= True, ret_value= 'verb'):

    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    with open(f"saved_features//{name}_RGB_{split}_{mode}.pkl", 'rb') as f:
        saved_features = pickle.load(f)

    with open(f"train_val//{split}_{mode}.pkl", 'rb') as f:
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
        label = labels[feat['uid']]
        
        for i in range(5):
            clip = feat['features_RGB'][i]
            clips_features.append(clip)
            clips_label.append(label)
            clips_narration.append(label['narration'])
            if cleanse and '-' in label['verb']:
                clips_verb.append(label['verb'].split('-')[0])
            else:
                clips_verb.append(label['verb'])
            clips_obj.append(label['narration'].split(' ')[-1])

    clips_features = np.array(clips_features)
    clips_narration = np.array(clips_narration)
    clips_obj = np.array(clips_obj)

    ## removing error classes (only one clip per class)

    if remove_errors:

        to_remove = []
        for v in set(clips_verb):
            if clips_verb.count(v) < (int(name.split('_')[0]) + 1):
                to_remove.append(v)
        print(f"removing {to_remove}")

        new_clips_features = []
        new_clips_narration = []
        new_clips_verb = []
        new_clips_obj = []
        for i in range(len(clips_features)):
            if clips_verb[i] not in to_remove:
                new_clips_features.append(clips_features[i])
                new_clips_narration.append(clips_narration[i])
                new_clips_verb.append(clips_verb[i])
                new_clips_obj.append(clips_obj[i])
        
        clips_features = np.array(new_clips_features)
        clips_narration = np.array(new_clips_narration)
        clips_verb = np.array(new_clips_verb)
        clips_obj = np.array(new_clips_obj)

    else:
        clips_verb = np.array(clips_verb)

    print(f"feature shape: {clips_features.shape}")
    print(f"labels shape: {clips_verb.shape}")

    if ret_value == 'narration':
        print(f"n classes (narrations): {len(set(clips_narration))}")
        print(f"{set(clips_narration)}")
        return clips_features, clips_narration
    
    elif ret_value == 'divided':
        print(f"n classes (narrations): {len(set(clips_narration))}")
        print(f"n classes (verbs): {len(set(clips_verb))}")
        print(f"{set(clips_verb)}")
        print(f"n classes (objs): {len(set(clips_obj))}")
        print(f"{set(clips_obj)}")
        return clips_features, clips_verb, clips_obj
    
    else:
        print(f"n classes (verbs): {len(set(clips_verb))}")
        print(f"{set(clips_verb)}")
        return clips_features, clips_verb
    

def scale_features(features, method= 'standard', ret_scaler= False):

    if method == 'standard':
        
        ## standard scaling

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if ret_scaler: return features_scaled, scaler
        else: return features_scaled

def get_colors(labels):

    label_set = set(labels)

    # Generate a unique color for each label
    cmap = [
        'black',
        'darkgrey',
        'rosybrown',
        'lightcoral',
        'firebrick',
        'orange',
        'sienna',
        'darkkhaki',
        'yellow',
        'greenyellow',
        'limegreen',
        'darkgreen',
        'lightseagreen',
        'turquoise',
        'darkslategrey',
        'steelblue',
        'mediumblue',
        'midnightblue',
        'slateblue',
        'rebeccapurple',
        'darkorchid',
        'plum',
        'pink'
#        "F0A3FF", #Amethyst
#        "0075DC", #Blue
#        "993F00", #Caramel
#        "4C005C", #Damson
#        "191919", #Ebony
#        "005C31", #Forest
#        "2BCE48", #Green
#        "FFCC99", #Honeydew
#        "808080", #Iron
#        "94FFB5", #Jade
#        "8F7C00", #Khaki
#        "9DCC00", #Lime
#        "C20088", #Mallow
#        "003380", #Navy
#        "FFA405", #Orpiment
#        "FFA8BB", #Pink
#        "426600", #Quagmire
#        "FF0010", #Red
#        "5EF1F2", #Sky
#        "00998F", #Turquoise
#        "E0FF66", #Uranium
#        "740AFF", #Violet
#        "990000", #Wine
#        "FFFF80", #Xanthin
#        "FFE100", #Yellow
#        "FF5005", #Zinnia
    ]

    if len(label_set) > len(cmap):
        cmap = plt.cm.get_cmap('viridis', len(label_set))
        cmap = [cmap(i) for i in range(len(label_set))]

    color_map = {label: cmap[i] for i, label in enumerate(set(labels))}
    colors = [color_map[label] for label in labels]

    return colors, color_map

def get_numerical_labels(labels):

    num_to_verb = list(set(labels))
    verb_to_num = {verb: i for i, verb in enumerate(num_to_verb)}
    numerical_labels = np.array([verb_to_num[verb] for verb in labels])

    return numerical_labels, num_to_verb, verb_to_num

def convert_to_numerical(labels, verb_to_num):
    return np.array([verb_to_num[verb] for verb in labels])