import pandas as pd
import numpy as np
import scipy
import scipy.signal as signal
from scipy.fftpack import fft
import math
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import librosa
import random
import librosa.display
import IPython.display
import pandas as pd
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
import warnings
from pydub import AudioSegment, effects
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import soundfile as sf

from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

def SpectralEntropy(signal, n_short_blocks=10):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + 0.00000001)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + 0.00000001))

    return entropy


def chromaImg(filename, path):

    print("reading " + filename + "...")

    songname = f'{path}/{filename}'

    if songname.endswith(".mp3"):
        mp3_audio = AudioSegment.from_file(songname, format="mp3")  # read mp3
        songname = mktemp('.wav')  # use temporary file
        mp3_audio.export(songname, format="wav")  # convert to wav

    rawsound = AudioSegment.from_file(songname, "wav")
    normalizedsound = effects.normalize(rawsound)
    songname = mktemp('.wav')  # use temporary file
    normalizedsound.export(songname, format="wav")

    y, sr = librosa.load(songname, mono=True)

    S = np.abs(librosa.stft(y, n_fft=2048))

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, sharex=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time')

    fig.colorbar(img)
    '''
    ax[0].label_outer()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])'''
    plt.show()

def PreProcessingFile(filename, path):
    print("reading " + filename + "...")

    songname = f'{path}/{filename}'

    if songname.endswith(".mp3"):
        mp3_audio = AudioSegment.from_file(songname, format="mp3")  # read mp3
        songname = mktemp('.wav')  # use temporary file
        mp3_audio.export(songname, format="wav")  # convert to wav

    rawsound = AudioSegment.from_file(songname, "wav")
    normalizedsound = effects.normalize(rawsound)
    songname = mktemp('.wav')  # use temporary file
    normalizedsound.export(songname, format="wav")

    y, sr = librosa.load(songname, mono=True)

    # First, design the Buterworth filter
    N = 1  # Filter order
    Wn = 0.9  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, y)
    plt.plot(y, 'r-')
    plt.plot(smooth_data, 'b-')
    plt.show()

    filename = 'result.wav'
    sf.write(filename, smooth_data, sr, format='wav')


def DecomposeTest(filename, path):
    print("reading " + filename + "...")

    songname = f'{path}/{filename}'

    if songname.endswith(".mp3"):
        mp3_audio = AudioSegment.from_file(songname, format="mp3")  # read mp3
        songname = mktemp('.wav')  # use temporary file
        mp3_audio.export(songname, format="wav")  # convert to wav

    rawsound = AudioSegment.from_file(songname, "wav")
    normalizedsound = effects.normalize(rawsound)
    songname = mktemp('.wav')  # use temporary file
    normalizedsound.export(songname, format="wav")

    y, sr = librosa.load(songname, mono=True)

    D = librosa.stft(y)
    rp = np.max(np.abs(D))

    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    D_harmonic2, D_percussive2 = librosa.decompose.hpss(D, margin=2)
    D_harmonic4, D_percussive4 = librosa.decompose.hpss(D, margin=4)
    D_harmonic8, D_percussive8 = librosa.decompose.hpss(D, margin=8)
    D_harmonic16, D_percussive16 = librosa.decompose.hpss(D, margin=16)

    """plt.figure(figsize=(10, 10))

    plt.subplot(5, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
    plt.title('Harmonic')
    plt.yticks([])
    plt.ylabel('margin=1')

    plt.subplot(5, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log')
    plt.title('Percussive')
    plt.yticks([]), plt.ylabel('')

    plt.subplot(5, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic2, ref=rp), y_axis='log')
    plt.yticks([])
    plt.ylabel('margin=2')

    plt.subplot(5, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive2, ref=rp), y_axis='log')
    plt.yticks([]), plt.ylabel('')

    plt.subplot(5, 2, 5)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic4, ref=rp), y_axis='log')
    plt.yticks([])
    plt.ylabel('margin=4')

    plt.subplot(5, 2, 6)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive4, ref=rp), y_axis='log')
    plt.yticks([]), plt.ylabel('')

    plt.subplot(5, 2, 7)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic8, ref=rp), y_axis='log')
    plt.yticks([])
    plt.ylabel('margin=8')

    plt.subplot(5, 2, 8)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive8, ref=rp), y_axis='log')
    plt.yticks([]), plt.ylabel('')

    plt.subplot(5, 2, 9)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic16, ref=rp), y_axis='log')
    plt.yticks([])
    plt.ylabel('margin=16')

    plt.subplot(5, 2, 10)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive16, ref=rp), y_axis='log')
    plt.yticks([]), plt.ylabel('')

    plt.tight_layout()
    plt.show()"""

    y_hat = librosa.istft(D_harmonic)
    filename = 'D_harmonic.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_percussive)
    filename = 'D_percussive.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_harmonic2)
    filename = 'D_harmonic2.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_percussive2)
    filename = 'D_percussive2.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_harmonic4)
    filename = 'D_harmonic4.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_percussive4)
    filename = 'D_percussive4.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_harmonic8)
    filename = 'D_harmonic8.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_percussive8)
    filename = 'D_percussive8.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_harmonic16)
    filename = 'D_harmonic16.wav'
    sf.write(filename, y_hat, sr, format='wav')
    y_hat = librosa.istft(D_percussive16)
    filename = 'D_percussive16.wav'
    sf.write(filename, y_hat, sr, format='wav')

def IRMASlabeling(path):
    for filename in os.listdir(f'{path}'):
        if filename.endswith(".txt"):
            with open(f'{path}/{filename}') as f:
                lines = f.readlines()
                print(lines)

"""path = 'C:/Users/Vano/Desktop/Diploma/База обучения'
filename = 'Labyrinth.wav'

DecomposeTest(filename, path)"""

path = 'C:/Users/Vano/Desktop/Diploma/База обучения'
filename = 'i-solisti-milano-angelo-ephrikian-allegro-for-organ-in-g-major.mp3'

chromaImg(filename, path)