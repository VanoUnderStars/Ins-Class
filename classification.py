import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import scipy
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, precision_recall_fscore_support
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import librosa.display
import os
import random
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import chardet
import warnings
warnings.filterwarnings('ignore')

instruments = 'Box Guitar Harmonica Mandolin Accordion Flute Organ Piano Violin Harpsichord'.split()
categories = 'Strings Wind Percussion'.split()
instruments_by_categories = ['Guitar Mandolin Piano Violin'.split(), 'Harmonica Flute Accordion Organ'.split(), 'Box Harpsichord'.split()]
UKP_instruments = "Banjo Bass_clairnet Bassoon Cello Clairnet Contrabasson Cor_anglias Double_bass Flute French_horn Guitar Mandolin Oboe Percussion Saxophone Trombone Trumpet Tuba Viola Violin".split()
UKPinstruments_by_categories = ['Banjo Cello Piano Violin Guitar Double_bass Mandolin Viola'.split(), 'Bassoon Bass_clairnet Clairnet Contrabasson Cor_anglias Flute French_horn Oboe Saxophone Trombone Trumpet Tuba'.split(), 'Percussion']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = 99
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def countSamples(y):
    for i in instruments:
        count = 0
        for j in range(len(y)):
            if y[j] == i:
                count = count + 1
        print(i)
        print(count)

def prepareData(file, dropFilename = True, testfile = False):
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())
    dataset = pd.read_csv(file, encoding=result['encoding'])
    dataset.head()
    dataset = dataset.reset_index()

    X = dataset
    if testfile == False:
        y = dataset['label']
        X = dataset.drop('label', 1)
    if dropFilename:
        X = X.drop('filename', 1)
    X = X.drop('index', 1)

    print(len(dataset))
    if testfile:
        return X, dataset
    else:
        return X, y, dataset

def internalRandomClassification(file):
    X, y, dataset = prepareData(file)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    splits = skf.split(X, y)
    i = 1

    parameters = {'max_depth': [1, 5, 10, 15, 20, 30, 35, 40, 50], 'n_estimators': [10, 20, 50, 100, 200, 250, 300, 350]}
    parameter_space = {
        'hidden_layer_sizes': [(100,100,100), (100,150,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [ 0.1, 0.5],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [2000]
    }
    etc = ExtraTreesClassifier()
    etc.get_params().keys()
    gcvNB = GridSearchCV(etc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcvNB.fit(X, y)
    print(gcvNB.cv_results_)
    print(gcvNB.best_estimator_)
    print(gcvNB.best_score_)


    etc = gcvNB.best_estimator_
    for train_index, test_index in splits:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        etc.fit(X_train, y_train)
        y_pred = etc.predict(X_test)

        print(accuracy_score(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

def classificateTestSong(train_file, test_file, classifier = ExtraTreesClassifier(max_depth=30, n_estimators=150)):

    X, y, dataset = prepareData(train_file)
    dataset_test = pd.read_csv(test_file)
    dataset_test.head()


    dataset_test = dataset_test.drop('filename', 1)
    X_test = dataset_test.drop('label', 1)

    classifier.fit(X, y)
    print(X_test)
    y_pred = classifier.predict(X_test)
    print(y_pred)
'''
testdataset = pd.read_csv('dataset_test.csv')
testdataset = testdataset.reset_index()

etc = ExtraTreesClassifier(n_estimators = 200)
etc.fit(X, y)
ytest = testdataset['label']
print(ytest)
testX = testdataset.drop('label', 1)
testX = testX.drop('filename', 1)
y_pred = etc.predict(testX)
plot_confusion_matrix(ytest, y_pred, instruments, normalize=False, title='Normalized confusion matrix')
plt.show()

testy = testdataset['label']
testname = testdataset['filename']
testX = testdataset.drop('label', 1)
testX = testX.drop('filename', 1)
y_pred = etc.predict(testX)
print(len(y))
print(len(testy))
print( accuracy_score(testy, y_pred))
print( precision_recall_fscore_support(testy, y_pred, average='macro'))
print( precision_recall_fscore_support(testy, y_pred, average='micro'))
uniquenames = []

for name in testname:
    uniquenames.append(name.split('.')[0])

unique = list(set(uniquenames))
percent = []

for name in unique:
    good = 0
    all = 0
    for i in range(len(testy)):
        if(testname[i].split('.')[0] == name):
            all = all + 1
            if(testy[i] == y_pred[i]):
                good = good + 1
    percent.append(good/all)

good = 0
all = 0
for i in range(len(percent)):
    all = all + 1
    if (percent[i] >= 0.50):
        good = good + 1
print(good / all)
'''
def featureImportance(file):
    X, y, dataset = prepareData(file)
    etc = ExtraTreesClassifier(max_depth=30, n_estimators=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    etc.fit(X_train, y_train)
    mfcc_n = 20
    header = 'rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tonnetz1 tonnetz2 tonnetz3 tonnetz4 tonnetz5 tonnetz6 energy spectral_flux super_flux strong_peak peak_prominences peak_widths chroma_mean'
    for i in range(1, mfcc_n + 1):
        header += f' mfcc{i}'
    feature_names = header.split()
    import time
    import numpy as np

    start_time = time.time()
    importances = etc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in etc.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Важность признаков по методу MDI")
    ax.set_ylabel("Среднее падение функции потерь")
    fig.tight_layout()

    start_time = time.time()
    result = permutation_importance(
        etc, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)


    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Важность признаков по методу Permutation Importance")
    ax.set_ylabel("Среднее уменьшение точности")
    fig.tight_layout()
    plt.show()


def classificateSongs(file):
    X, y, dataset = prepareData(file)
    filenames = dataset['filename']
    uniquenames = []
    IRMAS_instruments = "cel cla flu gac gel org pia sax tru vio voi".split()
    UKP_instruments = "Banjo Bass_clairnet Bassoon Cello Clairnet Contrabasson Cor_anglias Double_bass Flute French_horn Guitar Mandolin Oboe Percussion Saxophone Trombone Trumpet Tuba Viola Violin".split()
    for name in filenames:
        uniquenames.append(name.split('.')[0])
    unique = set(uniquenames)

    names_train = set(random.sample(list(unique), int(len(unique)*0.80)))
    names_test = unique - names_train


    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(len(filenames)):
        if(filenames[i].split('.')[0] in names_train):
            X_train.append(X.loc[i])
            y_train.append(y.loc[i])
        else:
            X_test.append(X.loc[i])
            y_test.append(y.loc[i])

    '''etc = ExtraTreesClassifier(max_depth=30, n_estimators=350)
    etc = RandomForestClassifier(max_depth=30, n_estimators=350)
    etc = MLPClassifier(alpha=0.1, hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive')
    SVC(kernel="linear", C=0.025),
    GaussianNB()
    DecisionTreeClassifier(max_depth=5),
    KNeighborsClassifier(3)'''
    etc = SVC(kernel="linear", C=0.025)
    start_time = time.time()


    etc.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to training: {elapsed_time:.3f} seconds")
    start_time = time.time()

    y_pred = etc.predict(X_test)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to predict: {elapsed_time:.3f} seconds")

    print(len(y_test))
    print( accuracy_score(y_test, y_pred))
    print( precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print( precision_recall_fscore_support(y_test, y_pred, average='micro'))

    '''plot_confusion_matrix(y_test, y_pred, instruments, normalize=True, title='Normalized confusion matrix')'''
    plt.show()

def treeStepClassification(file):
    X, y, dataset = prepareData(file)
    filenames = dataset['filename']
    uniquenames = []


    for name in filenames:
        uniquenames.append(name.split('.')[0])

    unique = set(uniquenames)

    y_cat = []
    for i in y:
        if i in instruments_by_categories[0]:
            y_cat.append('Strings')
        if i in instruments_by_categories[2]:
            y_cat.append('Percussion')
        if i in instruments_by_categories[1]:
            y_cat.append('Wind')


    names_train = set(random.sample(list(unique), int(len(unique)*0.80)))
    names_test = unique - names_train

    X_train, y_train, X_test, y_test, y_train_ins, y_test_ins = [], [], [], [], [], []

    for i in range(len(filenames)):
        if(filenames[i].split('.')[0] in names_train):
            X_train.append(X.loc[i])
            y_train_ins.append(y.loc[i])
            y_train.append(y_cat[i])
        else:
            X_test.append(X.loc[i])
            y_test_ins.append(y.loc[i])
            y_test.append(y_cat[i])


    etc = ExtraTreesClassifier(max_depth=30, n_estimators=200)
    etc.fit(X_train, y_train)
    y_pred = etc.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, categories, normalize=True, title='Normalized confusion matrix')
    plt.show()
    print(len(y_test))
    print(accuracy_score(y_test, y_pred))
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

    total_pred, total_test = [], []
    for i in range(len(categories)):
        X_train_cat, y_train_cat, X_test_cat, y_test_cat = [], [], [], []

        for j in range(len(X_train)):
            if y_train_ins[j] in instruments_by_categories[i]:
                X_train_cat.append(X_train[j])
                y_train_cat.append(y_train_ins[j])

        for j in range(len(y_pred)):
            if y_pred[j] == categories[i]:
                X_test_cat.append(X_test[j])
                y_test_cat.append(y_test_ins[j])

        etc.fit(X_train_cat, y_train_cat)
        y_pred_cat = etc.predict(X_test_cat)
        '''plot_confusion_matrix(y_test_cat, y_pred, instruments_by_categories[i], normalize=True, title='Normalized confusion matrix')
        plt.show()'''
        print(len(y_test_cat))
        print(accuracy_score(y_test_cat, y_pred_cat))
        print(precision_recall_fscore_support(y_test_cat, y_pred_cat, average='macro'))
        print(precision_recall_fscore_support(y_test_cat, y_pred_cat, average='micro'))

        for j in range(len(y_pred_cat)):
            total_pred.append(y_pred_cat[j])
            total_test.append(y_test_cat[j])

    print(len(total_test))
    print(accuracy_score(total_test, total_pred))
    print(precision_recall_fscore_support(total_test, total_pred, average='macro'))
    print(precision_recall_fscore_support(total_test, total_pred, average='micro'))


def treeStepClassificationTestSong(train_file, test_file):
    X, y, dataset = prepareData(train_file)
    filenames = dataset['filename']
    uniquenames = []


    for name in filenames:
        uniquenames.append(name.split('.')[0])

    unique = set(uniquenames)

    y_cat = []
    for i in y:
        if i in instruments_by_categories[0]:
            y_cat.append('Strings')
        if i in instruments_by_categories[2]:
            y_cat.append('Percussion')
        if i in instruments_by_categories[1]:
            y_cat.append('Wind')


    names_train = set(random.sample(list(unique), int(len(unique)*0.80)))
    names_test = unique - names_train

    X_train, y_train, X_test, y_test, y_train_ins, y_test_ins = [], [], [], [], [], []

    for i in range(len(filenames)):
        if(filenames[i].split('.')[0] in names_train):
            X_train.append(X.loc[i])
            y_train_ins.append(y.loc[i])
            y_train.append(y_cat[i])
        else:
            X_test.append(X.loc[i])
            y_test_ins.append(y.loc[i])
            y_test.append(y_cat[i])


    etc = ExtraTreesClassifier(max_depth=30, n_estimators=100)
    etc.fit(X_train, y_train)
    y_pred = etc.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, categories, normalize=True, title='Normalized confusion matrix')
    plt.show()
    print(len(y_test))
    print(accuracy_score(y_test, y_pred))
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

    total_pred, total_test = [], []
    for i in range(len(categories)):
        X_train_cat, y_train_cat, X_test_cat, y_test_cat = [], [], [], []

        for j in range(len(X_train)):
            if y_train_ins[j] in instruments_by_categories[i]:
                X_train_cat.append(X_train[j])
                y_train_cat.append(y_train_ins[j])

        for j in range(len(y_pred)):
            if y_pred[j] == categories[i]:
                X_test_cat.append(X_test[j])
                y_test_cat.append(y_test_ins[j])

        etc.fit(X_train_cat, y_train_cat)
        y_pred_cat = etc.predict(X_test_cat)
        '''plot_confusion_matrix(y_test_cat, y_pred, instruments_by_categories[i], normalize=True, title='Normalized confusion matrix')
        plt.show()'''
        print(len(y_test_cat))
        print(accuracy_score(y_test_cat, y_pred_cat))
        print(precision_recall_fscore_support(y_test_cat, y_pred_cat, average='macro'))
        print(precision_recall_fscore_support(y_test_cat, y_pred_cat, average='micro'))

        for j in range(len(y_pred_cat)):
            total_pred.append(y_pred_cat[j])
            total_test.append(y_test_cat[j])

    print(len(total_test))
    print(accuracy_score(total_test, total_pred))
    print(precision_recall_fscore_support(total_test, total_pred, average='macro'))
    print(precision_recall_fscore_support(total_test, total_pred, average='micro'))

def classificateIRMAS(file, testfile, testlabelfile):
    X, y, dataset = prepareData(file)
    X_test, testdataset = prepareData(testfile, dropFilename = True, testfile = True)
    filenames = testdataset['filename']

    IRMAS_instruments = "cel cla flu gac gel org pia sax tru vio voi".split()

    with open(testlabelfile, 'rb') as f:
        result = chardet.detect(f.read())

    labeldataset = pd.read_csv(testlabelfile, encoding=result['encoding'])
    labeldataset.head()
    labeldataset = labeldataset.reset_index()

    y_test = []

    for filename in filenames:
        y_test.append(labeldataset[(labeldataset['filename'] + '.wav') == filename]['label'])


    etc = ExtraTreesClassifier(max_depth=30, n_estimators=200)
    etc.fit(X, y)

    y_pred = etc.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, IRMAS_instruments, normalize=True, title='Normalized confusion matrix')
    plt.show()

    print(len(y_test))
    print( accuracy_score(y_test, y_pred))
    print( precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print( precision_recall_fscore_support(y_test, y_pred, average='micro'))
'''percent = []

for name in names_test:
    good = 0
    all = 0
    for i in range(len(y_test)):
        if(testname[i].split('.')[0] == name):
            all = all + 1
            if(y_test[i] == y_pred[i]):
                good = good + 1
    percent.append(good/all)

good = 0
all = 0
for i in range(len(percent)):
    all = all + 1
    if (percent[i] >= 0.50):
        good = good + 1
print(good / all)'''

'''internalRandomClassification(IRMAS_train)
classificateTestSong('dataset_opti_test.csv', 'dataset_file.csv')

treeStepClassification('dataset_opti_test.csv')
classificateSongs('IRMAS_train.csv')
featureImportance('IRMAS_train.csv')

classificateSongs('UKP_dataset.csv')

classificateIRMAS('IRMAS_train.csv', 'IRMAS_test_Part1.csv', 'IRMAS_test_Part1_labels.csv')

treeStepClassification('dataset_opti.csv')'''
classificateTestSong('dataset_opti.csv', 'dataset_file.csv')