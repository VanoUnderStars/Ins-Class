import pandas as pd
import numpy as np
import scipy
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


from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")

instruments_list_short = 'Box Guitar Harmonica Mandolin Accordion Piano'
instruments_list = 'Box Guitar Harmonica Mandolin Accordion Flute Organ Piano Violin Harpsichord'
instruments_list_long = 'Box Guitar Harmonica Mandolin Accordion Flute Organ Piano Violin Harpsichord'

mfcc_n = 20
header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tonnetz1 tonnetz2 tonnetz3 tonnetz4 tonnetz5 tonnetz6 energy spectral_flux super_flux strong_peak peak_prominences peak_widths chroma_mean'
for i in range(1, mfcc_n + 1):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

def distribution_shape( y ):
    centralMoments = []
    d_shape = []
    for i in range(0, 5):
        centralMoments.append(scipy.stats.moment(y, moment=i))
    d_shape.append(np.var(centralMoments))
    d_shape.append(scipy.stats.skew(centralMoments))
    d_shape.append(scipy.stats.kurtosis(centralMoments))
    return d_shape

'''def SpectralFlux(y, window):
    # compute the spectral flux as the sum of square distances:
    sp_flux = []
    current_position = 0
    while current_position + window - 1 < len(y):
        y_frame = y[current_position:current_position + window]
        fft_magnitude = abs(fft(y_frame))
        if current_position != 0:
            fft_sum = np.sum(fft_magnitude + 0.00000001)
            previous_fft_sum = np.sum(fft_magnitude_previous + 0.00000001)
            sp_flux.append(np.sum((fft_magnitude / fft_sum - fft_magnitude_previous /previous_fft_sum) ** 2))
        fft_magnitude_previous = fft_magnitude.copy()
        current_position = current_position + window
    return np.mean(sp_flux)'''

def StrongPeak(y):
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.5)
    prominences = peak_prominences(y, peaks)[0]
    max = 0
    maxi = 0
    for i in range(0, prominences.size):
        if (prominences[i] > max):
            max = prominences[i]
            maxi = i
    return (max/results_half[0][maxi])

def PeaksPower(y):
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.5)
    prominences = peak_prominences(y, peaks)[0]
    return np.average(prominences)

def PeaksPower2(y):
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.5)
    prominences = peak_prominences(y, peaks)[0]
    return np.average(results_half)

def PeaksPower3(y):
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    peaks, _ = find_peaks(y)
    results_half = peak_widths(y, peaks, rel_height=0.5)
    prominences = peak_prominences(y, peaks)[0]
    return np.average(np.true_divide(prominences, results_half))

def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s))
    return entropy


def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)

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


def harmonic(frame, sampling_rate):
    """
    Computes harmonic ratio and pitch
    """
    m = np.round(0.016 * sampling_rate) - 1
    r = np.correlate(frame, frame, mode='full')

    g = r[len(frame) - 1]
    r = r[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(r)))

    if len(a) == 0:
        m0 = len(r) - 1
    else:
        m0 = a[0]
    if m > len(r):
        m = len(r) - 1

    gamma = np.zeros(int(m), dtype=np.float64)
    cumulative_sum = np.cumsum(frame ** 2)
    gamma[m0:int(m)] = r[m0:int(m)] / (np.sqrt((g * cumulative_sum[int(m):m0:-1])))

    zcr = zero_crossing_rate(gamma)

    if zcr > 0.15:
        hr = 0.0
        f0 = 0.0
    else:
        if len(gamma) == 0:
            hr = 1.0
            blag = 0.0
            gamma = np.zeros((m), dtype=np.float64)
        else:
            hr = np.max(gamma)
            blag = np.argmax(gamma)

        # Get fundamental frequency:
        f0 = sampling_rate / (blag)
        if f0 > 5000:
            f0 = 0.0
        if hr < 0.1:
            f0 = 0.0

    return hr, f0

def random_list(window, frames_choosed, size, y, silence_energy_range):
    list = []
    i = 0

    if size < frames_choosed:
        frames_choosed = size

    j = 0

    while i < frames_choosed and j < size:
        r = random.randint(0, size)
        if r not in list:
            y_frame = y[r*window:r*window + window]
            en = energy(y_frame)
            if (np.mean(en) > silence_energy_range):
                list.append(r)
                i = i + 1
        j = j + 1

    return list

def preProcessFile(songname):
    if songname.endswith(".mp3"):
        mp3_audio = AudioSegment.from_file(songname, format="mp3")  # read mp3
        songname = mktemp('.wav')  # use temporary file
        mp3_audio.export(songname, format="wav")  # convert to wav

    rawsound = AudioSegment.from_file(songname, "wav")
    normalizedsound = effects.normalize(rawsound)
    songname = mktemp('.wav')  # use temporary file
    normalizedsound.export(songname, format="wav")

    return songname

def getParametersFromFrame(y_frame, sr):
    rmse = librosa.feature.rms(y=y_frame, frame_length=len(y_frame))
    chroma_stft = librosa.feature.chroma_stft(y=y_frame, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y_frame, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_frame, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_frame, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y_frame)
    tonnetz = librosa.feature.tonnetz(y=y_frame, sr=sr)
    mfcc = librosa.feature.mfcc(y=y_frame, sr=sr, n_mfcc=20)
    s_flux = librosa.onset.onset_strength(y=y_frame, sr=sr,
                                          hop_length=int(librosa.time_to_samples(1. / 200, sr=sr)))
    sup_flux = librosa.onset.onset_detect(y=y_frame, sr=sr,
                                          hop_length=int(librosa.time_to_samples(1. / 200, sr=sr)),
                                          units='time')
    entropy = SpectralEntropy(y_frame)
    to_append = f' {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tonnetz[0])} {np.mean(tonnetz[1])} {np.mean(tonnetz[2])} {np.mean(tonnetz[3])} {np.mean(tonnetz[4])} {np.mean(tonnetz[5])} {np.mean(entropy)} {np.mean(s_flux)} {np.mean(sup_flux)} {np.mean(StrongPeak(y_frame))} {np.mean(PeaksPower(y_frame))} {np.mean(PeaksPower2(y_frame))} {np.mean(chroma_stft)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    return to_append

"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, sharex=True)
times = librosa.times_like(par)
ax.semilogy(np.arange(len(en)) * window_sec, en, Label = filename)
ax.set(xticks=[])
ax.legend()
ax.label_outer()
plt.show()"""


def createBase(path):
    mfcc_n = 20
    header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tonnetz1 tonnetz2 tonnetz3 tonnetz4 tonnetz5 tonnetz6 energy spectral_flux strong_peak peak_prominences peak_widths chroma_stft'
    for i in range(1, mfcc_n + 1):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    # We write the data to a csv file

    file = open('dataset_test.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    instruments = instruments_list.split()
    dur = []
    for g in instruments:
        files = 0
        for filename in os.listdir(f'{path}/{g}'):
            print("reading " + filename + "...")
            songname = f'{path}/{g}/{filename}'

            if songname.endswith(".mp3"):
                mp3_audio = AudioSegment.from_file(songname, format="mp3")  # read mp3
                songname = mktemp('.wav')  # use temporary file
                mp3_audio.export(songname, format="wav")  # convert to wav

            rawsound = AudioSegment.from_file(songname, "wav")
            normalizedsound = effects.normalize(rawsound)
            songname = mktemp('.wav')  # use temporary file
            normalizedsound.export(songname, format="wav")

            y, sr = librosa.load(songname, mono=True)
            silence_energy_range = 0.001

            window_sec = 5
            window = sr * window_sec
            current_position = 0
            num = 0
            file = open('dataset_test.csv', 'a', newline='')
            files = files + 1
            '''if files>50:
                break'''

            while current_position + window - 1 < len(y):
                y_frame = y[current_position:current_position + window]
                rmse = librosa.feature.rms(y=y_frame, frame_length=len(y_frame))
                chroma_stft = librosa.feature.chroma_stft(y=y_frame, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y_frame, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y_frame, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y_frame, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y_frame)
                tonnetz = librosa.feature.tonnetz(y_frame, sr=sr)
                mfcc = librosa.feature.mfcc(y=y_frame, sr=sr, n_mfcc=mfcc_n)
                flatness = librosa.feature.spectral_flatness(y=y_frame)
                fft_magnitude = abs(fft(y_frame))
                s_flux = librosa.onset.onset_strength(y=y_frame, sr=sr,
                                                           hop_length=int(librosa.time_to_samples(1. / 200, sr=sr)))
                sup_flux = librosa.onset.onset_detect(y=y_frame, sr=sr,
                                                           hop_length=int(librosa.time_to_samples(1. / 200, sr=sr)),
                                                           units='time')
                s_entropy = SpectralEntropy(fft_magnitude)
                """h, p = harmonic(y_frame, sr)"""
                entropy = SpectralEntropy(y_frame)
                en = energy(y_frame)
                current_position = current_position + window
                num = num + 1
                name = filename + '_' + f'{num}'
                if (np.mean(en) > silence_energy_range):
                    to_append = f'{name} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tonnetz[0])} {np.mean(tonnetz[1])} {np.mean(tonnetz[2])} {np.mean(tonnetz[3])} {np.mean(tonnetz[4])} {np.mean(tonnetz[5])} {np.mean(entropy)} {np.mean(s_flux)} {np.mean(sup_flux)} {np.mean(StrongPeak(y))} {np.mean(PeaksPower(y))} {np.mean(PeaksPower2(y))} {np.mean(chroma_stft)}'
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    to_append += f' {g}'
                    file = open('dataset_test.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

def createOptimizedBase(path, window_sec = 5, samples_num = 10, silence_energy_range = 0.001):

    # We write the data to a csv file
    file = open('dataset_opti.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    instruments = instruments_list.split()
    for g in instruments:
        print("read " + g + " samples:")
        for filename in os.listdir(f'{path}/{g}'):
            print("read " + filename + "...")
            songname = f'{path}/{g}/{filename}'

            songname = preProcessFile(songname)

            y, sr = librosa.load(songname, mono=True)

            window = sr * window_sec
            list = random_list(window, samples_num, int(len(y)/window), y, silence_energy_range)
            for i in list:
                y_frame = y[i*window:i*window + window]
                to_append = f'{filename}'
                to_append += getParametersFromFrame(y_frame, sr)
                to_append += f' {g}'
                file = open('dataset.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

def createTestCsvFromFile(filename, path, window_sec = 5, silence_energy_range = 0.001):
    print("read " + filename + "...")

    songname = f'{path}/{filename}'
    songname = preProcessFile(songname)

    y, sr = librosa.load(songname, mono=True)

    window = sr * window_sec
    current_position = 0
    num = 0
    file = open('dataset_file.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    while current_position + window - 1 < len(y):
        y_frame = y[current_position:current_position + window]
        en = energy(y_frame)
        current_position = current_position + window
        num = num + 1
        name = filename + '_' + f'{num}'
        if (np.mean(en) > silence_energy_range):
            to_append = f'{name}'
            to_append += getParametersFromFrame(y_frame, sr)
            file = open('dataset_file.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def createIRMASdata(path):

    file = open('IRMAS_train.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    IRMAS_instruments = "cel cla flu gac gel org pia sax tru vio voi".split()
    for g in IRMAS_instruments:
        print("reading " + g + " samples:")
        for filename in os.listdir(f'{path}/{g}'):
            print("reading " + filename + "...")
            songname = f'{path}/{g}/{filename}'

            songname = preProcessFile(songname)

            y, sr = librosa.load(songname, mono=True)
            silence_energy_range = 0.001

            window_sec = 1
            window = sr * window_sec
            current_position = 0
            num = 0
            while current_position + window - 1 < len(y):
                y_frame = y[current_position:current_position + window]
                en = energy(y_frame)
                current_position = current_position + window
                num = num + 1
                name = filename + '_' + f'{num}'
                if (np.mean(en) > silence_energy_range):
                    to_append = f'{name}'
                    to_append += getParametersFromFrame(y_frame, sr)
                    to_append += f' {g}'
                    file = open('IRMAS_train.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

def createIRMAStest(path):
    header.remove('label')
    file = open('IRMAS_test_Part1.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for filename in os.listdir(f'{path}'):
        print("reading " + filename + "...")
        songname = f'{path}/{filename}'

        songname = preProcessFile(songname)

        y, sr = librosa.load(songname, mono=True)
        silence_energy_range = 0.001

        window_sec = 1
        window = sr * window_sec
        current_position = 0

        while current_position + window - 1 < len(y):
            y_frame = y[current_position:current_position + window]
            en = energy(y_frame)
            current_position = current_position + window
            if (np.mean(en) > silence_energy_range):
                to_append = f'{filename}'
                to_append += getParametersFromFrame(y_frame, sr)
                file = open('IRMAS_test_Part1.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

def createIRMAStestLabels(path):
    file = open('IRMAS_test_Part1_labels.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow('filename label'.split())
    for filename in os.listdir(f'{path}'):
        if filename.endswith(".txt"):
            with open(f'{path}/{filename}') as f:
                lines = f.readlines()
                name = filename.replace('.txt','')
                to_append = f'{name}'
                label = lines[0].replace('\t\n','')
                print(label)
                to_append += f' {label}'
                file = open('IRMAS_test_Part1_labels.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())


path = 'C:/Users/Vano/Desktop/Diploma/База обучения'
filename = 'dirge_for_the_planet.wav'

"""createOptimizedBase(path)"""

createTestCsvFromFile(filename, path)
