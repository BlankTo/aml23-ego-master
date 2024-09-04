import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import pandas as pd

import librosa
import matplotlib.pyplot as plt

def compute_spectrogram(readings, n_fft= 32, win_length= None, hop_length= 4):

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        normalized=True,
    )

    return torch.stack([spectrogram(readings[:, j]) for j in range(8)])

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", show= True):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    if show: plt.show()

if __name__ == "__main__":

    #spec = pd.read_pickle('saved_features/actionEMGspec_train.pkl')
    #print(spec.shape)
    #print(spec.columns)
    #print(spec['left_spectrogram'].shape)
    #print(spec['left_spectrogram'].iloc[0].shape)
    #
    #plot_spectrogram(spec['left_spectrogram'].iloc[0], show= False)

    data = pd.read_pickle('saved_features/actionEMG_train.pkl')

    #left_spectrograms = []
    #right_spectrograms = []
    #for i in range(len(data)):
    #    print(f'\r{i+1}/{len(data)}', end='', flush=True)
    #    left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"]))
    #
    #print()
    #plot_spectrogram(left_spectrograms[0], show= False)
    #
    #left_spectrograms = []
    #right_spectrograms = []
    #for i in range(len(data)):
    #    print(f'\r{i+1}/{len(data)}', end='', flush=True)
    #    left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], n_fft= 64))
    #
    #print()
    #plot_spectrogram(left_spectrograms[0], show= False)
    #
    #left_spectrograms = []
    #right_spectrograms = []
    #for i in range(len(data)):
    #    print(f'\r{i+1}/{len(data)}', end='', flush=True)
    #    left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], n_fft= 8))
    #
    #print()
    #plot_spectrogram(left_spectrograms[0], show= False)
    #
    #left_spectrograms = []
    #right_spectrograms = []
    #for i in range(len(data)):
    #    print(f'\r{i+1}/{len(data)}', end='', flush=True)
    #    left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], n_fft= 64, hop_length= 2))
    #
    #print()
    #plot_spectrogram(left_spectrograms[0], show= False)
    #
    #left_spectrograms = []
    #right_spectrograms = []
    #for i in range(len(data)):
    #    print(f'\r{i+1}/{len(data)}', end='', flush=True)
    #    left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], hop_length= 2))
    #
    #print()
    #plot_spectrogram(left_spectrograms[0], show= False)

    left_spectrograms = []
    right_spectrograms = []
    for i in range(len(data)):
        print(f'\r{i+1}/{len(data)}', end='', flush=True)
        left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], hop_length= 1, n_fft= 16))

    print()
    plot_spectrogram(left_spectrograms[0], show= False)

    left_spectrograms = []
    right_spectrograms = []
    for i in range(len(data)):
        print(f'\r{i+1}/{len(data)}', end='', flush=True)
        left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], hop_length= 1))

    print()
    plot_spectrogram(left_spectrograms[0], show= False)

    left_spectrograms = []
    right_spectrograms = []
    for i in range(len(data)):
        print(f'\r{i+1}/{len(data)}', end='', flush=True)
        left_spectrograms.append(compute_spectrogram(data.iloc[i]["myo_left_readings"], hop_length= 1, n_fft= 64))

    print()
    plot_spectrogram(left_spectrograms[0])