import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_features_RGB(name, split= 'D1', mode= 'train'):

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
    clips_class = []
    for feat in features:
        label = labels[feat['uid']]
        
        for i in range(5):
            clip = feat['features_RGB'][i]
            clips_features.append(clip)
            clips_class.append(label['verb_class'])

    clips_features = np.array(clips_features)
    clips_class = np.array(clips_class)

    print(f"feature shape: {clips_features.shape}")
    print(f"labels shape: {clips_class.shape}")

    return clips_features, clips_class
    

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