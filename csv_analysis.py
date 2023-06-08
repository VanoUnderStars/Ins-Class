import pandas
import matplotlib.pyplot as plt
import numpy as np


instruments = 'Box Guitar Harmonica Mandolin Accordion Flute Organ Piano Violin Harpsichord'.split()
df = pandas.read_csv('dataset_opti_test.csv')


plt.rcdefaults()

mfcc_n = 20
header = 'rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tonnetz1 tonnetz2 tonnetz3 tonnetz4 tonnetz5 tonnetz6 energy spectral_flux strong_peak peak_prominences peak_widths chroma_stft'
for i in range(1, mfcc_n + 1):
    header += f' mfcc{i}'
header = header.split()

release_header = ['RMS', 'Spectral Centroid',  'Spectral Bandwidth',  'Spectral Rolloff', 'Zero Crossing Rate', 'Tonnetz 1', 'Tonnetz 2', 'Tonnetz 3','Tonnetz 4','Tonnetz 5', 'Tonnetz6',  'Energy', 'Spectral Flux',  'Strong Peak',  'Peak Prominences',  'Peak Widths', 'Chroma STFT']
for i in range(1, 21):
    release_header.append(f'MFCC {i}')


for par in range(0, len(header)):
    fig, ax = plt.subplots()
    value = header[par]

    y_pos = np.arange(len(instruments))
    performance = []
    for i in range(0,10):
        result = 0.0
        div = 0
        for j in range(0, len(df)):
            if df.get('label')[j] == instruments[i]:
                result = result + df.get(value)[j]
                div = div + 1
        performance.append(result/div)

    print(performance)
    ax.barh(y_pos, np.array(performance), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(instruments)
    ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_xlabel(value)
    ax.set_title(release_header[par])
    mng = plt.get_current_fig_manager()
    plt.savefig(f'{value}.png')


def showStd(df, value = 'flatness'):
    stdmas = []
    meanmas = []

    for i in range(0,9):
        min = 1000000
        mas = []
        for j in range(0, len(df)):
            if df.get('label')[j] == instruments[i]:
                mas.append(df.get(value)[j])
                if (df.get(value)[j] < min):
                    min = df.get(value)[j]
        if(min < 0):
            mas = mas + abs(min)
        stdmas.append(np.std(mas)/np.mean(mas))
        meanmas.append(np.mean(mas))

    print(value)
    print(str(np.mean(stdmas)) + ' ' + str(np.std(meanmas)/np.mean(meanmas)))


def showAll(df, value = 'flatness'):
    performance = []
    names = []
    max = -10000
    maxname = minname = 'None'
    min = 10000
    fig, ax = plt.subplots()


    for j in range(0, len(df)):
        #if df.get('label')[j] == instruments[instrument_num]:
        element = df.get(value)[j]
        if(element < min):
            min = element
            minname = df.get('filename')[j]
        if (element > max):
            max = element
            maxname = df.get('filename')[j]
        performance.append(element)
        names.append(df.get('filename')[j])


    y_pos = np.arange(len(performance))
    ax.barh(y_pos, np.array(performance),  align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(value)

    print('Max: ' + maxname + ' Min: ' + minname)

    plt.show()

'''for value in header:
    showStd(df=df, value = value)

showAll(df, value = 'dshape1')'''