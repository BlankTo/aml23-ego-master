import torch
import torch
import torchaudio.transforms as T
import torch
import numpy as np
from scipy.signal import butter, filtfilt

import pandas as pd
import numpy as np

import librosa
import matplotlib.pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(
            librosa.power_to_db(specgram[i]), origin="lower", aspect="auto"
        )
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show(block=False)


def preprocess(readings):
    # * apply preprocessing to the EMG data

    # * Rectification
    # abs value
    readings_rectified = np.abs(readings)
    # * low-pass Filter
    # Frequenza di campionamento (Hz)
    fs = 160  # Frequenza dei sampling data da loro
    f_cutoff = 5  # Frequenza di taglio

    # Ordine del filtro
    order = 4
    # Calcolo dei coefficienti del filtro
    b, a = butter(order, f_cutoff / (fs / 2), btype="low")
    # Concateno tutti i vettori in un'unica matrice
    readings_filtered = np.zeros_like(readings_rectified, dtype=float)
    for i in range(8):  # 8 colonne
        readings_filtered[:, i] = filtfilt(b, a, readings_rectified[:, i])

    # print(readings_rectified[:6], readings_rectified.shape)
    # print(readings_filtered[:6], readings_filtered.shape)
    # exit()

    # convert to tensor
    readings_filtered = torch.tensor(readings_filtered, dtype=torch.float32)

    min_val, _ = torch.min(readings_filtered, dim=1, keepdim=True)
    max_val, _ = torch.max(readings_filtered, dim=1, keepdim=True)

    g = max_val - min_val + 0.0001

    # # Normalize the data to the range -1 to 1
    normalized_data = 2 * (readings_filtered - min_val) / g - 1

    return normalized_data


def compute_spectrogram(readings):

    # Sampling frequency is 160 Hz
    # With 32 samples the frequency resolution after FFT is 160 / 32
    n_fft = 32
    win_length = None
    hop_length = 4

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        normalized=True,
    )

    signal = [[float(x) for x in record] for record in readings]
    signal = torch.tensor(signal, dtype=torch.float32)
    freq_signal = torch.stack([spectrogram(readings[:, j]) for j in range(8)])

    return freq_signal


def compute_spectrogram_alt(readings):
    # Sampling frequency is 160 Hz
    # With 32 samples the frequency resolution after FFT is 160 / 32
    n_fft = 32
    win_length = None
    hop_length = 4

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        normalized=True,
    )

    for i in range(len(readings)):
        signal_l = (readings[i]["left_readings"]).float()
        signal_r = (readings[i]["right_readings"]).float()

        freq_signal_l = [spectrogram(signal_l[:, j]) for j in range(8)]
        freq_signal_r = [spectrogram(signal_r[:, j]) for j in range(8)]

        readings[i]["left_readings"] = freq_signal_l
        readings[i]["right_readings"] = freq_signal_r