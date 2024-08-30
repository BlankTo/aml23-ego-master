import random
import pickle
import statistics
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import torch
from .spec_emg import compute_spectrogram


def pkl_to_pd(pkl_file_path):
    with open(pkl_file_path, "rb") as pkl_file:
        data = pd.read_pickle(pkl_file)
    return data

label_dict = {
    "Spread jelly on a bread slice": 0,
    "Slice a potato": 1,
    "Get items from refrigerator/cabinets/drawers": 2,
    "Get/replace items from refrigerator/cabinets/drawers": 2,
    "Clean a plate with a towel": 3,
    "Pour water from a pitcher into a glass": 4,
    "Stack on table: 3 each large/small plates, bowls": 5,
    "Spread almond butter on a bread slice": 6,
    "Slice a cucumber": 7,
    "Clean a pan with a sponge": 8,
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 9,
    "Open a jar of almond butter": 10,
    "Open/close a jar of almond butter": 10,
    "Slice bread": 11,
    "Peel a cucumber": 12,
    "Clean a plate with a sponge": 13,
    "Clear cutting board": 14,
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 15,
    "Clean a pan with a towel": 16,
    "Peel a potato": 17,
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 18,
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": 19,
}

fps = 30
offset = 5  # seconds
labels = [
    "Spread jelly on a bread slice",
    "Slice a potato",
    "Get/replace items from refrigerator/cabinets/drawers",
    "Clean a plate with a towel",
    "Pour water from a pitcher into a glass",
    "Stack on table: 3 each large/small plates, bowls",
    "Spread almond butter on a bread slice",
    "Slice a cucumber",
    "Clean a pan with a sponge",
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Open/close a jar of almond butter",
    "Slice bread",
    "Peel a cucumber",
    "Clean a plate with a sponge",
    "Clear cutting board",
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Clean a pan with a towel",
    "Peel a potato",
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
]

labels_remapping = {
    "Spread jelly on a bread slice": (0, "Spread"),
    "Slice a potato": (1, "Slice"),
    "Get/replace items from refrigerator/cabinets/drawers": (2, "Get/Put"),
    "Clean a plate with a towel": (3, "Clean"),
    "Pour water from a pitcher into a glass": (4, "Pour"),
    "Stack on table: 3 each large/small plates, bowls": (5, "Stack"),
    "Spread almond butter on a bread slice": (0, "Spread"),
    "Slice a cucumber": (1, "Slice"),
    "Clean a pan with a sponge": (3, "Clean"),
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        6,
        "Load",
    ),
    "Open/close a jar of almond butter": (7, "Open/Close"),
    "Slice bread": (1, "Slice"),
    "Peel a cucumber": (1, "Slice"),
    "Clean a plate with a sponge": (3, "Clean"),
    "Clear cutting board": (3, "Clean"),
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        5,
        "Stack",
    ),
    "Clean a pan with a towel": (3, "Clean"),
    "Peel a potato": (1, "Slice"),
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        6,
        "Load",
    ),
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils": (
        2,
        "Get/Put",
    ),
}

uid = random.randint(0, 20000)
dataset = []
dataset_reduced = []
dataset_emg = []
timestamps = []
timestamps_int = []


def generate_record(index, row, first_frame, cnt=1, emg=False):
    record = {}
    record["uid"] = uid + index
    record["participant_id"] = "P04"
    record["video_id"] = f"P04_0{cnt}"
    if not emg:
        if row["description"] == "Get items from refrigerator/cabinets/drawers":
            description = "Get/replace items from refrigerator/cabinets/drawers"
        if row["description"] == "Open a jar of almond butter":
            description = "Open/close a jar of almond butter"
    record["narration"] = row["description"]
    record["verb"] = row["description"]
    if emg:
        record["verb_class"] = label_dict[row["description"]]
    else:
        record["verb_class"] = labels.index(row["description"])
    record["start_timestamp"] = float(row["start"])
    record["stop_timestamp"] = float(row["stop"])
    record["start_frame"] = round((float(row["start"]) - first_frame) * fps)
    record["stop_frame"] = round((float(row["stop"]) - first_frame) * fps)
    record["myo_right_timestamps"] = row["myo_right_timestamps"]
    record["myo_left_timestamps"] = row["myo_left_timestamps"]
    record["myo_right_readings"] = row["myo_right_readings"]
    record["myo_left_readings"] = row["myo_left_readings"]

    return record


def dataset_augmentation(record, uid):
    duration = int(record["stop_timestamp"] - record["start_timestamp"])
    if duration < (offset * 2):
        return [record]

    records = []
    next = 0
    first_iteration = True
    while first_iteration or duration - next > offset:
        first_iteration = False
        new_record = record.copy()

        new_record["uid"] = uid
        new_record["start_timestamp"] = new_record["start_timestamp"] + next
        new_record["stop_timestamp"] = new_record["start_timestamp"] + offset

        new_record["start_frame"] = (
            new_record["start_frame"] + (fps * next) - (21 if next > 0 else 0)
        )
        new_record["stop_frame"] = new_record["start_frame"] + (fps * offset)

        next += offset + 1
        uid += 1
        if duration - next < offset:
            new_record["stop_timestamp"] = record["stop_timestamp"]
            new_record["stop_frame"] = record["stop_frame"]

        readings = []
        for i in range(len(new_record["myo_left_timestamps"])):
            ts = new_record["myo_left_timestamps"][i]
            if int(ts) >= int(new_record["start_timestamp"]) and int(ts) <= int(
                new_record["stop_timestamp"]
            ):
                readings.append(new_record["myo_left_readings"][i])
        new_record["myo_left_readings"] = readings

        readings = []
        for i in range(len(new_record["myo_right_timestamps"])):
            ts = new_record["myo_right_timestamps"][i]
            if int(ts) >= int(new_record["start_timestamp"]) and int(ts) <= int(
                new_record["stop_timestamp"]
            ):
                readings.append(new_record["myo_right_readings"][i])
        new_record["myo_right_readings"] = readings
        records.append(new_record)

    return records


def remap_labels(record):
    record_reduced = record.copy()
    record_reduced["verb_class"] = labels_remapping[record["verb"]][0]
    record_reduced["verb"] = labels_remapping[record["verb"]][1]
    return record_reduced


def sampling(readings):
    for i in range(2, 11):
        value = 750 * (10 / i)
        if len(readings) >= value:
            r_cp = []
            for k in range(0, int(value), 10):
                r_cp.extend(readings[k : k + i])
            return r_cp[:750]


def emg_dataset(path, out_path):
    actionNet_train = pkl_to_pd(path)
    data = {"features": []}
    dataset_emg = []
    uid_offset = 0
    first_frame = 0

    for i in range(len(actionNet_train)):
        index = actionNet_train.index[i]
        file = actionNet_train.iloc[i].file
        label = actionNet_train.iloc[i].description

        file_pkl = pkl_to_pd("action_net/action_net_dataset/" + file)
        row = file_pkl.loc[index]
        first_frame = float(file_pkl.loc[0]["start"])
        record = generate_record(uid_offset, row, first_frame, 1, True)

        records = dataset_augmentation(record, uid_offset)
        uid_offset += len(records)
        for r in records:
            dataset_emg.append(
                {
                    "id": r["uid"],
                    "right_readings": r["myo_right_readings"],
                    "left_readings": r["myo_left_readings"],
                    "label": label_dict[label],
                }
            )

        for i in range(len(dataset_emg)):
            readings = dataset_emg[i]["right_readings"]
            if len(readings) > 750:
                readings = sampling(readings)
            elif len(readings) < 750:
                new_rows = np.zeros((750 - len(readings), 8))
                readings = np.concatenate((readings, new_rows), axis=0)
            dataset_emg[i]["right_readings"] = readings

        for i in range(len(dataset_emg)):
            readings = dataset_emg[i]["left_readings"]
            if len(readings) > 750:
                readings = sampling(readings)
            elif len(readings) < 750:
                new_rows = np.zeros((750 - len(readings), 8))
                readings = np.concatenate((readings, new_rows), axis=0)
            dataset_emg[i]["left_readings"] = readings

    data["features"] = dataset_emg
    with open(out_path, "wb") as file:
        pickle.dump(data, file)

    print("EMG Action Net Creation: done")
    return data


def preprocess(readings):
    #* apply preprocessing to the EMG data

    #* Rectification
    # abs value
    readings_rectified = np.abs(readings)
    #* low-pass Filter
    # Frequenza di campionamento (Hz)
    fs = 160  # Frequenza dei sampling data da loro
    f_cutoff = 5  # Frequenza di taglio

    # Ordine del filtro
    order = 4 
    # Calcolo dei coefficienti del filtro
    b, a = butter(order, f_cutoff / (fs / 2), btype='low')
    # Concateno tutti i vettori in un'unica matrice
    readings_filtered = np.zeros_like(readings_rectified, dtype=float)
    for i in range(8):  # 8 colonne
        readings_filtered[:, i] = filtfilt(b, a, readings_rectified[:, i])

    #print(readings_rectified[:6], readings_rectified.shape)
    #print(readings_filtered[:6], readings_filtered.shape)
    # exit()

    # convert to tensor
    readings_filtered = torch.tensor(readings_filtered, dtype=torch.float32)
    
    min_val, _ = torch.min(readings_filtered, dim=1, keepdim=True)
    max_val, _ = torch.max(readings_filtered, dim=1, keepdim=True)

    g = max_val - min_val + 0.0001

    # # Normalize the data to the range -1 to 1
    normalized_data = 2 * (readings_filtered - min_val) / g - 1

    #print(normalized_data[:6], normalized_data.shape)


    return normalized_data


def emg_dataset_spettrogram(path, out_path):
    actionNet_train = pkl_to_pd(path)
    data = {"features": []}
    dataset_final = []
    uid_offset = 0
    first_frame = 0

    for i in range(len(actionNet_train)):
        index = actionNet_train.index[i]
        file = actionNet_train.iloc[i].file
        label = actionNet_train.iloc[i].description
        dataset_emg = []
        dataset_spect = []

        file_pkl = pkl_to_pd("action_net/pickles/" + file)
        row = file_pkl.loc[index]
        first_frame = float(file_pkl.loc[0]["start"])
        record = generate_record(uid_offset, row, first_frame, 1, True)

        records = dataset_augmentation(record, uid_offset)
        uid_offset += len(records)
        for r in records:
            dataset_emg.append(
                {
                    "id": r["uid"],
                    "right_readings": r["myo_right_readings"],
                    "left_readings": r["myo_left_readings"],
                    "label": label_dict[label],
                }
            )

        for k in range(len(dataset_emg)):
            readings = dataset_emg[k]["right_readings"]
            if len(readings) > 750:
                readings = sampling(readings)
            elif len(readings) < 750:
                new_rows = np.zeros((750 - len(readings), 8))
                readings = np.concatenate((readings, new_rows), axis=0)
            dataset_emg[k]["right_readings"] = preprocess(readings)

        for k in range(len(dataset_emg)):
            readings = dataset_emg[k]["left_readings"]
            if len(readings) > 750:
                readings = sampling(readings)
            elif len(readings) < 750:
                new_rows = np.zeros((750 - len(readings), 8))
                readings = np.concatenate((readings, new_rows), axis=0)
            dataset_emg[k]["left_readings"] = preprocess(readings)

        for k in range(len(dataset_emg)):
            dataset_spect.append(
                {
                    "id": dataset_emg[k]["id"],
                    "right_readings": compute_spectrogram(dataset_emg[k]["right_readings"]),
                    "left_readings": compute_spectrogram(dataset_emg[k]["left_readings"]),
                    "label": dataset_emg[k]["label"],
                }
            )
        dataset_final.extend(dataset_spect)

        print(f"Spectogramm: {i}/{len(actionNet_train)}")
        
    data["features"] = dataset_final
    with open(out_path, "wb") as file:
        pickle.dump(data, file)

    print("EMG_comv Action Net Creation: done")
    return data


def emg_analysis(folder):
    dataset_emg = []
    for file in folder:
        print(file)
        data = pkl_to_pd(file)
        uid_offset = 0
        first_frame = 0

        for index, row in data.iterrows():
            if index == 0:
                first_frame = float(row["start"])
                continue

            record = generate_record(uid_offset, row, first_frame, 1)

            # Dataset augmentation by splitting video
            records = dataset_augmentation(record, uid_offset)
            uid_offset += len(records)
            for r in records:
                dataset_emg.append(
                    {
                        "id": r["uid"],
                        "right_readings": r["myo_right_readings"],
                        "left_readings": r["myo_left_readings"],
                        "label": r["verb_class"],
                    }
                )

    for i in range(len(dataset_emg)):
        readings = dataset_emg[i]["right_readings"]
        if len(readings) > 750:
            readings = sampling(readings)
        elif len(readings) < 750:
            new_rows = np.zeros((750 - len(readings), 8))
            readings = np.concatenate((readings, new_rows), axis=0)
        dataset_emg[i]["right_readings"] = readings

    for i in range(len(dataset_emg)):
        readings = dataset_emg[i]["left_readings"]
        if len(readings) > 750:
            readings = sampling(readings)
        elif len(readings) < 750:
            new_rows = np.zeros((750 - len(readings), 8))
            readings = np.concatenate((readings, new_rows), axis=0)
        dataset_emg[i]["left_readings"] = readings

    # Convert list of dictionaries to DataFrame
    df_emg = pd.DataFrame(dataset_emg)

    print("EMG Action Net Creation: done")
    return df_emg


def rgb_action_net_creation(out_path=None, out_path_reduced=None, out_path_emg=None):
    data = pkl_to_pd("action_net/pickles/S04_1.pkl")
    uid_offset = 0
    first_frame = 0
    dataset_spect = []

    for index, row in data.iterrows():
        if index == 0:
            first_frame = float(row["start"])
            continue

        record = generate_record(uid_offset, row, first_frame, 1)

        # Dataset augmentation by splitting video
        records = dataset_augmentation(record, uid_offset)
        uid_offset += len(records)
        for r in records:
            # Adding record(s) to dataset
            r_rgb = {
                "uid": r["uid"],
                "participant_id": r["participant_id"],
                "video_id": r["video_id"],
                "narration": r["narration"],
                "verb": r["verb"],
                "verb_class": r["verb_class"],
                "start_timestamp": r["start_timestamp"],
                "stop_timestamp": r["stop_timestamp"],
                "start_frame": r["start_frame"],
                "stop_frame": r["stop_frame"],
            }

            dataset.append(r_rgb)

            dataset_emg.append(
                {
                    "id": r["uid"],
                    "right_readings": r["myo_right_readings"],
                    "left_readings": r["myo_left_readings"],
                    "label": r["verb_class"],
                }
            )

            # Remap labels
            record_reduced = remap_labels(r_rgb)
            dataset_reduced.append(record_reduced)

    # EMG adjustments
    for i in range(len(dataset_emg)):
        readings = dataset_emg[i]["right_readings"]
        if len(readings) > 750:
            readings = sampling(readings)
        elif len(readings) < 750:
            new_rows = np.zeros((750 - len(readings), 8))
            readings = np.concatenate((readings, new_rows), axis=0)
        dataset_emg[i]["right_readings"] = preprocess(readings)

    for i in range(len(dataset_emg)):
        readings = dataset_emg[i]["left_readings"]
        if len(readings) > 750:
            readings = sampling(readings)
        elif len(readings) < 750:
            new_rows = np.zeros((750 - len(readings), 8))
            readings = np.concatenate((readings, new_rows), axis=0)
        dataset_emg[i]["left_readings"] = preprocess(readings)

    for k in range(len(dataset_emg)):
        print(f"Spectogramm: {k}/{len(dataset_emg)}")
        dataset_spect.append(
            {
                "id": dataset_emg[k]["id"],
                "right_readings": compute_spectrogram(dataset_emg[k]["right_readings"]),
                "left_readings": compute_spectrogram(dataset_emg[k]["left_readings"]),
                "label": dataset_emg[k]["label"],
            }
        )

    # Convert list of dictionaries to DataFrame
    df_rgb = pd.DataFrame(dataset)
    df_reduced = pd.DataFrame(dataset_reduced)
    df_emg = pd.DataFrame(dataset_spect)

    # Split dataset into train and validation
    train_data, val_data = train_test_split(df_rgb, test_size=0.2, random_state=42)
    train_data_emg, val_data_emg = train_test_split(
        df_emg, test_size=0.2, random_state=42
    )
    train_data_reduced, val_data_reduced = train_test_split(
        df_reduced, test_size=0.2, random_state=42
    )

    if out_path is not None:
        # Save numpy array to .pkl file
        with open(f"{out_path}_train.pkl", "wb") as file:
            pickle.dump(train_data, file)

        with open(f"{out_path}_test.pkl", "wb") as file:
            pickle.dump(val_data, file)

    if out_path_reduced is not None:
        # Save numpy array to .pkl file (reduced labels)
        with open(f"{out_path_reduced}_train.pkl", "wb") as file:
            pickle.dump(train_data_reduced, file)

        with open(f"{out_path_reduced}_test.pkl", "wb") as file:
            pickle.dump(val_data_reduced, file)

    if out_path_emg is not None:
        # Save numpy array to .pkl file
        with open(f"{out_path_emg}_train.pkl", "wb") as file:
            pickle.dump(train_data_emg, file)

        with open(f"{out_path_emg}_test.pkl", "wb") as file:
            pickle.dump(val_data_emg, file)

    print("RGB Action Net Creation: done")
    return df_rgb, df_reduced, df_emg
