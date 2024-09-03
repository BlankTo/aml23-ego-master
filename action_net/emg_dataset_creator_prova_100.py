import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import random as rand
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

from spec import compute_spectrogram

relabel_dict = {'Open a jar of almond butter': 'Open/close a jar of almond butter',
                'Get items from refrigerator/cabinets/drawers': 'Get/replace items from refrigerator/cabinets/drawers'}

def pkl_to_pd(pkl_file_path):
    with open(pkl_file_path, "rb") as pkl_file:
        data = pd.read_pickle(pkl_file)
    return data

def create_emg_datasets(dataset_folder, clip_duration= 5, fps= 30):

    data = None
    uid = 0

    for pkl_file in os.listdir(dataset_folder):
        print(pkl_file)

        new_data = pkl_to_pd(os.path.join(dataset_folder, pkl_file))
        new_data['participant_id'] = pkl_file.split('_')[0]
        new_data['video_id'] = pkl_file.split('.pkl')[0]
        new_data = new_data[new_data['description'] != 'calibration']
        first_frame = new_data['start'].iloc[0]

        for index, row in new_data.iterrows():

            record = {}
            if row['description'] in relabel_dict.keys(): row['description'] = relabel_dict[row['description']]

            des = row['description'].lower()
            if 'spread' in des: record['verb_class'] = 'Spread'
            elif 'slice' in des or 'peel' in des: record['verb_class'] = 'Slice'
            elif 'get' in des or 'put' in des or 'replace' in des: record['verb_class'] = 'Get/Put'
            elif 'clean' in des or 'clear' in des: record['verb_class'] = 'Clean'
            elif 'pour' in des: record['verb_class'] = 'Pour'
            elif 'stack' in des or 'set' in des: record['verb_class'] = 'Stack'
            elif 'load' in des: record['verb_class'] = 'Load'
            elif 'open' in des or 'close' in des: record['verb_class'] = 'Open/Close'
            else:
                print(f'no label matching -> {des}')
                exit()

            record['verb'] = des.split(' ')[0]

            record['narration'] = row['description']

            record['start_timestamp'] = row['start']
            record['start_frame'] = round((row['start'] - first_frame) * fps)

            record['stop_timestamp'] = row['stop']
            record['stop_frame'] = round((row['stop'] - first_frame) * fps)

            record['myo_left_timestamps'] = row['myo_left_timestamps']
            record['myo_left_readings'] = row['myo_left_readings']
            record['myo_right_timestamps'] = row['myo_right_timestamps']
            record['myo_right_readings'] = row['myo_right_readings']

            sample_duration = int(record["stop_timestamp"] - record["start_timestamp"])

            if sample_duration < (clip_duration * 2):

                ##

                record['uid'] = uid
                uid += 1

                left_readings = []
                for i in range(len(record["myo_left_timestamps"])):
                    ts = record["myo_left_timestamps"][i]
                    if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                        left_readings.append(record["myo_left_readings"][i])

                if len(left_readings) > 100:
                    i = math.ceil(100 * 10 / len(left_readings))
                    value = 100 * (10 / i)
                    r_cp = []
                    for k in range(0, int(value), 10):
                        r_cp.extend(left_readings[k : k + i])
                    r_cp = r_cp[:100]
                    left_readings = np.array(r_cp)

                elif len(left_readings) < 100:
                    new_rows = np.zeros((100 - len(left_readings), 8))
                    left_readings = np.concatenate((left_readings, new_rows), axis=0)

                left_readings_rectified = np.abs(left_readings)
                fs = 160
                f_cutoff = 5
                order = 4
                b, a = butter(order, f_cutoff / (fs / 2), btype= 'low')
                left_readings_filtered = np.zeros_like(left_readings_rectified, dtype= float)
                for i in range(8):
                    left_readings_filtered[:, i] = filtfilt(b, a, left_readings_rectified[:, i])

                left_readings_filtered = torch.tensor(left_readings_filtered, dtype= torch.float32)
                
                min_val, _ = torch.min(left_readings_filtered, dim=1, keepdim=True)
                max_val, _ = torch.max(left_readings_filtered, dim=1, keepdim=True)

                g = max_val - min_val + 0.0001

                normalized_left_readings = 2 * (left_readings_filtered - min_val) / g - 1

                record["myo_left_readings"] = normalized_left_readings

                ##

                #right_readings = []
                #for i in range(len(record["myo_right_timestamps"])):
                #    ts = record["myo_right_timestamps"][i]
                #    if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                #        right_readings.append(record["myo_right_readings"][i])
                #clip["myo_right_readings"] = right_readings

                right_readings = []
                for i in range(len(record["myo_right_timestamps"])):
                    ts = record["myo_right_timestamps"][i]
                    if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                        right_readings.append(record["myo_right_readings"][i])

                if len(right_readings) > 100:
                    i = math.ceil(100 * 10 / len(right_readings))
                    value = 100 * (10 / i)
                    r_cp = []
                    for k in range(0, int(value), 10):
                        r_cp.extend(right_readings[k : k + i])
                    r_cp = r_cp[:100]
                    right_readings = np.array(r_cp)

                elif len(right_readings) < 100:
                    new_rows = np.zeros((100 - len(right_readings), 8))
                    right_readings = np.concatenate((right_readings, new_rows), axis=0)

                right_readings_rectified = np.abs(right_readings)
                fs = 160
                f_cutoff = 5
                order = 4
                b, a = butter(order, f_cutoff / (fs / 2), btype= 'low')
                right_readings_filtered = np.zeros_like(right_readings_rectified, dtype= float)
                for i in range(8):
                    right_readings_filtered[:, i] = filtfilt(b, a, right_readings_rectified[:, i])

                right_readings_filtered = torch.tensor(right_readings_filtered, dtype= torch.float32)
                
                min_val, _ = torch.min(right_readings_filtered, dim=1, keepdim=True)
                max_val, _ = torch.max(right_readings_filtered, dim=1, keepdim=True)

                g = max_val - min_val + 0.0001

                normalized_right_readings = 2 * (right_readings_filtered - min_val) / g - 1

                record["myo_right_readings"] = normalized_right_readings

                ##

                records = [record]
            else:
                records = []
                clip_start = 0
                while clip_start + clip_duration < sample_duration:
                    clip = {}

                    clip['uid'] = uid
                    uid += 1

                    clip['verb'] = record['verb']
                    clip['verb_class'] = record['verb_class']
                    clip['narration'] = record['narration']

                    clip['start_timestamp'] = record['start_timestamp'] + clip_start
                    clip['stop_timestamp'] = clip['start_timestamp'] + clip_duration

                    clip['start_frame'] = record['start_frame'] + (fps * clip_start)
                    clip['stop_frame'] = clip['start_frame'] + (fps * clip_duration)

                    clip_start += clip_duration + 1

                    if clip_start + clip_duration < sample_duration:
                        clip['stop_timestamp'] = record['stop_timestamp']

                    ##

                    left_readings = []
                    for i in range(len(record["myo_left_timestamps"])):
                        ts = record["myo_left_timestamps"][i]
                        if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                            left_readings.append(record["myo_left_readings"][i])

                    if len(left_readings) > 100:
                        i = math.ceil(100 * 10 / len(left_readings))
                        value = 100 * (10 / i)
                        r_cp = []
                        for k in range(0, int(value), 10):
                            r_cp.extend(left_readings[k : k + i])
                        r_cp = r_cp[:100]
                        left_readings = np.array(r_cp)

                    elif len(left_readings) < 100:
                        new_rows = np.zeros((100 - len(left_readings), 8))
                        left_readings = np.concatenate((left_readings, new_rows), axis=0)

                    left_readings_rectified = np.abs(left_readings)
                    fs = 160
                    f_cutoff = 5
                    order = 4
                    b, a = butter(order, f_cutoff / (fs / 2), btype= 'low')
                    left_readings_filtered = np.zeros_like(left_readings_rectified, dtype= float)
                    for i in range(8):
                        left_readings_filtered[:, i] = filtfilt(b, a, left_readings_rectified[:, i])

                    left_readings_filtered = torch.tensor(left_readings_filtered, dtype= torch.float32)
                    
                    min_val, _ = torch.min(left_readings_filtered, dim=1, keepdim=True)
                    max_val, _ = torch.max(left_readings_filtered, dim=1, keepdim=True)

                    g = max_val - min_val + 0.0001

                    normalized_left_readings = 2 * (left_readings_filtered - min_val) / g - 1

                    clip["myo_left_readings"] = normalized_left_readings

                    ##

                    right_readings = []
                    for i in range(len(record["myo_right_timestamps"])):
                        ts = record["myo_right_timestamps"][i]
                        if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                            right_readings.append(record["myo_right_readings"][i])

                    if len(right_readings) > 100:
                        i = math.ceil(100 * 10 / len(right_readings))
                        value = 100 * (10 / i)
                        r_cp = []
                        for k in range(0, int(value), 10):
                            r_cp.extend(right_readings[k : k + i])
                        r_cp = r_cp[:100]
                        right_readings = np.array(r_cp)

                    elif len(right_readings) < 100:
                        if False: continue #ppp
                        new_rows = np.zeros((100 - len(right_readings), 8))
                        right_readings = np.concatenate((right_readings, new_rows), axis=0)

                    right_readings_rectified = np.abs(right_readings)
                    fs = 160
                    f_cutoff = 5
                    order = 4
                    b, a = butter(order, f_cutoff / (fs / 2), btype= 'low')
                    right_readings_filtered = np.zeros_like(right_readings_rectified, dtype= float)
                    for i in range(8):
                        right_readings_filtered[:, i] = filtfilt(b, a, right_readings_rectified[:, i])

                    right_readings_filtered = torch.tensor(right_readings_filtered, dtype= torch.float32)
                    
                    min_val, _ = torch.min(right_readings_filtered, dim=1, keepdim=True)
                    max_val, _ = torch.max(right_readings_filtered, dim=1, keepdim=True)

                    g = max_val - min_val + 0.0001

                    normalized_right_readings = 2 * (right_readings_filtered - min_val) / g - 1

                    clip["myo_right_readings"] = normalized_right_readings

                    ##

                    records.append(clip)

            if data is None:
                data = pd.DataFrame(records)
            else:
                data = pd.concat([data, pd.DataFrame(records)], ignore_index=True)

    label_set = set(data['verb_class'].to_list())
    label_dict = {lab: i for i, lab in enumerate(label_set)}
    print(label_dict)
    labels = []
    for _, row in data.iterrows():
        labels.append(label_dict[row['verb_class']])
    data['label'] = labels

    verb_set = set(data['verb'].to_list())
    verb_dict = {lab: i for i, lab in enumerate(verb_set)}
    print(verb_dict)
    verbs = []
    for _, row in data.iterrows():
        verbs.append(verb_dict[row['verb']])
    data['verb_label'] = verbs

    action_set = set(data['narration'].to_list())
    action_dict = {lab: i for i, lab in enumerate(action_set)}
    print(action_dict)
    actions = []
    for _, row in data.iterrows():
        actions.append(action_dict[row['narration']])
    data['action_label'] = actions

    left_spectrograms = []
    right_spectrograms = []
    for i in range(len(data)):
        print(f'\r{i+1}/{len(data)}', end='', flush=True)
        left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], hop_length= 1))
        right_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_right_readings"], hop_length= 1))

    data['left_spectrogram'] = left_spectrograms
    data['right_spectrogram'] = right_spectrograms

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    rgb_train_data = train_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame']]
    with open(f'train_val/action_{clip_duration}s_100_RGB_train.pkl', 'wb') as out_file:
        pickle.dump(rgb_train_data, out_file)

    rgb_test_data = test_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame']]
    with open(f'train_val/action_{clip_duration}s_100_RGB_test.pkl', 'wb') as out_file:
        pickle.dump(rgb_test_data, out_file)

    emg_train_data = train_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'myo_left_timestamps', 'myo_left_readings', 'myo_right_timestamps', 'myo_right_readings']]
    with open(f'saved_features/action_{clip_duration}s_100_EMG_train.pkl', 'wb') as out_file:
        pickle.dump(emg_train_data, out_file)

    emg_test_data = test_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'myo_left_timestamps', 'myo_left_readings', 'myo_right_timestamps', 'myo_right_readings']]
    with open(f'saved_features/action_{clip_duration}s_100_EMG_test.pkl', 'wb') as out_file:
        pickle.dump(emg_test_data, out_file)

    spec_train_data = train_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'left_spectrogram', 'right_spectrogram']]
    with open(f'saved_features/action_{clip_duration}s_100_EMGspec_train.pkl', 'wb') as out_file:
        pickle.dump(spec_train_data, out_file)

    spec_test_data = test_data[['uid', 'verb', 'verb_class', 'label', 'action_label', 'verb_label', 'narration', 'start_timestamp', 'stop_timestamp', 'left_spectrogram', 'right_spectrogram']]
    with open(f'saved_features/action_{clip_duration}s_100_EMGspec_test.pkl', 'wb') as out_file:
        pickle.dump(spec_test_data, out_file)


###################

dataset_folder = "action_net/action_net_dataset"

#######

create_emg_datasets(dataset_folder, clip_duration= 5)

#######

create_emg_datasets(dataset_folder, clip_duration= 10)