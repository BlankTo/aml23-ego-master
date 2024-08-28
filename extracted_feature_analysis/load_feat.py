import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_features_RGB(name, split= 'D1', mode= 'train', return_one_clip= False, return_central_frame= False):

    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    with open(name, 'rb') as f:
        saved_features = pickle.load(f)

    with open(f"train_val//{split}_{mode}.pkl", 'rb') as f:
        train_val = pickle.load(f)

    labels = {train_val.iloc[i]['uid']: {'narration': train_val.iloc[i]['narration'], 'verb': train_val.iloc[i]['verb'], 'verb_class': train_val.iloc[i]['verb_class']} for i in range(len(train_val))}

    features = saved_features['features']
    central_frames = saved_features['central_frames']
    clips_features = []
    clips_class = []
    clips_central_frame = []
    for (feat, cf) in zip(features, central_frames):
        label = labels[feat['uid']]

        if return_one_clip:
            clip = feat['features_RGB'][0]
            clips_features.append(clip)
            clips_class.append(label['verb_class'])
            clips_central_frame.append(cf)

        else:
        
            for i in range(5):
                clip = feat['features_RGB'][i]
                clips_features.append(clip)
                clips_class.append(label['verb_class'])
                clips_central_frame.append(cf)

    clips_features = np.array(clips_features)
    clips_class = np.array(clips_class)
    clips_central_frame = np.array(clips_central_frame)

    #print(f"feature shape: {clips_features.shape}")
    #print(f"labels shape: {clips_class.shape}")

    if return_central_frame:
        #print(f"central_frames shape: {clips_class.shape}")
        return clips_features, clips_class, clips_central_frame
    
    return clips_features, clips_class


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
        'pink',
    ]

    if len(label_set) > len(cmap):
        cmap = plt.cm.get_cmap('viridis', len(label_set))
        cmap = [cmap(i) for i in range(len(label_set))]

    color_map = {label: cmap[i] for i, label in enumerate(set(labels))}
    colors = [color_map[label] for label in labels]

    return colors, color_map