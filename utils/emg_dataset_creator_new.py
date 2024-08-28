import os
import math
import torch
import numpy as np
import pandas as pd
import random as rand
from scipy.signal import butter, filtfilt

def pkl_to_pd(pkl_file_path):
    with open(pkl_file_path, "rb") as pkl_file:
        data = pd.read_pickle(pkl_file)
    return data

label_dict = {}

fps = 30
clip_duration = 5

labels = []

labels_remapping = {}

uid = rand.randint(0, 20000)
dataset = []
dataset_reduced = []
dataset_emg = []
timestamps = []
timestamps_int = []

dataset_folder = "action-net/action_net_dataset"

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

                #left_readings = []
                #for i in range(len(record["myo_left_timestamps"])):
                #    ts = record["myo_left_timestamps"][i]
                #    if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                #        left_readings.append(record["myo_left_readings"][i])
                #clip["myo_left_readings"] = left_readings

                left_readings = []
                for i in range(len(record["myo_left_timestamps"])):
                    ts = record["myo_left_timestamps"][i]
                    if int(ts) >= int(record["start_timestamp"]) and int(ts) <= int(record["stop_timestamp"]):
                        left_readings.append(record["myo_left_readings"][i])

                if len(left_readings) > 750:
                    i = math.ceil(750 * 10 / len(left_readings))
                    value = 750 * (10 / i)
                    r_cp = []
                    for k in range(0, int(value), 10):
                        r_cp.extend(left_readings[k : k + i])
                    r_cp = r_cp[:750]
                    left_readings = r_cp

                elif len(left_readings) < 750:
                    new_rows = np.zeros((750 - len(left_readings), 8))
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

                if len(right_readings) > 750:
                    i = math.ceil(750 * 10 / len(right_readings))
                    value = 750 * (10 / i)
                    r_cp = []
                    for k in range(0, int(value), 10):
                        r_cp.extend(right_readings[k : k + i])
                    r_cp = r_cp[:750]
                    right_readings = r_cp

                elif len(right_readings) < 750:
                    new_rows = np.zeros((750 - len(right_readings), 8))
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


        ## data aug

        if data is None:
            data = pd.DataFrame(records)
        else:
            data = pd.concat([data, pd.DataFrame(records)], ignore_index=True)

print(data.shape)
print(data.columns)
print(set(data['verb_class'].to_list()))
print(len(set(data['verb_class'].to_list())))

rgb_data = data[['uid', 'verb', 'verb_class', 'narration', 'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame']]
emg_data = data[['uid', 'verb', 'verb_class', 'narration', 'start_timestamp', 'stop_timestamp', 'myo_left_timestamps', 'myo_left_readings', 'myo_right_timestamps', 'myo_right_readings']]


left_spectrograms = []
right_spectrograms = []
for i in range(len(data)):
    left_spectrograms.append(compute_spectrogram(data[i]["left_readings"]))
    right_spectrograms.append(compute_spectrogram(data[i]["right_readings"]))

exit()

#TODO why this sampling?

for i in range(len(myo_data)):
    right_readings = myo_data[i]["right_readings"]

    if len(right_readings) > 750:
        i = math.ceil(750 * 10 / len(right_readings))
        value = 750 * (10 / i)
        r_cp = []
        for k in range(0, int(value), 10):
            r_cp.extend(right_readings[k : k + i])
        r_cp = r_cp[:750]
        right_readings = r_cp

    elif len(right_readings) < 750:
        new_rows = np.zeros((750 - len(right_readings), 8))
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

    myo_data[i]["right_readings"] = normalized_right_readings

for i in range(len(myo_data)):
    left_readings = myo_data[i]["left_readings"]

    if len(left_readings) > 750:
        i = math.ceil(750 * 10 / len(left_readings))
        value = 750 * (10 / i)
        r_cp = []
        for k in range(0, int(value), 10):
            r_cp.extend(left_readings[k : k + i])
        r_cp = r_cp[:750]
        left_readings = r_cp

    elif len(left_readings) < 750:
        new_rows = np.zeros((750 - len(left_readings), 8))
        left_readings = np.concatenate((left_readings, new_rows), axis=0)

    ##
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

    myo_data[i]["left_readings"] = normalized_left_readings

emg_dataset = pd.DataFrame(emg_data)