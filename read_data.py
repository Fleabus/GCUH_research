import wfdb
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from plot_signal import plot_signal

#Categories 17
abnormal = ['A']
categories = ['N','A']
colours = ['#3cb44b', '#e6194b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6'
            ,'#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000']

#HyperParameters
signalType = "V1"
hz = 900
downsample = 300
SecondsWanted = 2500 #Seconds of data wanted
dynamicPeak = False # If true, the ecg slice will pick based on if the highest or lowest point is greater
peakSelection = 1 # 1 == pick highest point. -1 == pick lowest point. This only works if dynamic peak is false
min_val = -1 # Used for normalization
max_val = 1 # Used for normalization
offsetMax = 0 # Used for shifting the data

#Global Variables
signalIndex = 1
sampSize = int(SecondsWanted*hz) #Sets sample size to number of seconds by Hz(360)
sampIncrement = 1/hz
annotationArray = []
signalArray = []


#Initially read data in
def readData(filename):
    global annotationArray
    global signalArray
    #read in data using
    '''
    record = wfdb.rdsamp(filename, sampto = sampSize)
    annotation = wfdb.rdann(filename, 'atr', sampto = sampSize)
    sig, fields = wfdb.srdsamp(filename, sampto = sampSize)
    '''
    record = wfdb.rdsamp(filename)
    annotation = wfdb.rdann(filename, 'atr')
    sig, fields = wfdb.srdsamp(filename)
    print("\nReading in ", len(sig)/300, "seconds of data from file", filename)

    for i in range(0, len(sig)):
        if(i%100 == 0):
            sys.stdout.write("\rReading Data ... {0:.2f}%".format((float(i)/len(sig))*100))
            sys.stdout.flush()
        signalArray.append(sig[i])

    sys.stdout.write("\rReading Data ... complete!")
    sys.stdout.flush()

    for i in range(1, len(annotation.annsamp)):
        annotationArray.append([annotation.annsamp[i], annotation.anntype[i]])


def sort_annotations():
    for i in range(len(annotationArray)):
        try:
            annotationArray[i-1][1] = np.identity(len(categories))[categories.index(annotationArray[i-1][1])]
        except ValueError:
            if(annotationArray[i-1][1] in abnormal):
                annotationArray[i-1][1] = np.identity(len(categories))[1]
            else:
                annotationArray[i-1][1] = [-1]

def slice_annotations(signals, annotations):
    x = []
    y = []
    for label in annotations:
        index = label[0]
        label = label[1]

        high = int(index + hz/2)
        low = int(index - hz/2)

        if(high > len(signals)):
            high = len(signals)
        if(low < 0):
            low = 0

'''
Signals two-tuple [[sig1, sig2], [sig1, sig2] ... [sig1, sig2]]
Annoations two-tuple [[index, type], [index, type] ... [index, type]]

Find highest peak of signal within 300hz of annotation index
Slice peak 300hz either way
'''
def slice_peaks(signals, annotations, peak=False):
        # Lists to contain new values
        x = []
        y = []
        for label in annotations:

            tempLabel = []
            assign = True
            index = label[0]
            label = label[1]

            high = int(index + hz/2)
            low = int(index - hz/2)

            if(high > len(signals)):
                high = len(signals)
            if(low < 0):
                low = 0
            # slice range from annotation index
            signalRange = [signals[i][signalIndex] for i in range(low, high)]

            # find highest peak index
            maxIndex = label[0]
            minIndex = label[0]
            average = np.mean(signalRange)
            if(maxIndex < 0):
                maxIndex = 0
            #average = 0
            if(abs(signalRange[int(maxIndex)] - average) < abs(signalRange[int(minIndex)] - average)):
                peakIndex = minIndex
            else:
                peakIndex = maxIndex
            if(dynamicPeak == False):
                if(peakSelection == 1):
                    peakIndex = maxIndex
                if(peakSelection == -1):
                    peakIndex = minIndex

            peakIndex += low
            if(peak == False):
                peakIndex = index
            high = int(peakIndex + hz/2)
            low = int(peakIndex - hz/2)

            if(high+offsetMax > len(signals)):
                assign = False
            if(low-offsetMax < 0):
                assign = False
            if(label[0] == -1):
                assign = False
            if assign:
                signalRange = signals[low:high]
                x.append([i[signalIndex] for i in signalRange])
                y.append(label)

        # Assign lists to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y


def plotSignal(features, labels, alpha_value=0.5):
    plt.style.use('dark_background')
    for i in range(len(features)):
        #plt.scatter(range(len(features[i])), features[i],c=range(len(features[i])), marker='_', s=1)
        plt.plot(features[i], color=colours[np.argmax(labels[i])], alpha=alpha_value)


# returns the two arrays
def retrieveData():
    return signalArray, annotationArray


def binarySegment(x, y):
    x_norm = []
    y_norm = []
    x_abnorm = []
    y_abnorm = []

    for i in range(len(x)):
        if(y[i][0] == 1):
            x_norm.append(x[i])
            y_norm.append(y[i])
        else:
            x_abnorm.append(x[i])
            y_abnorm.append(y[i])
    x_norm = np.array(x_norm)
    y_norm = np.array(y_norm)
    x_abnorm = np.array(x_abnorm)
    y_abnorm = np.array(y_abnorm)
    return x_norm, y_norm, x_abnorm, y_abnorm

def calculateDeltaChange(data):
    new_data = []
    for x in data:
        new_snippet = []
        for i in range(len(x)):
            if i == 0:
                new_snippet.append(0)
            else:
                new_snippet.append(x[i-1]-x[i])
        new_data.append(new_snippet)
    return np.array(new_data)


'''
Retrieves all data based on parameters specified
params:
    - sigType = "MLII"
    - directory = "mitdb"
'''
def loadAllData(sigType="MLII", directory="mitdb"):
    signalType = sigType
    for file in os.listdir(directory):
        if file.endswith(".dat"):
            sig, fields = wfdb.srdsamp(directory + "/" + os.path.splitext(file)[0])
            if(fields['signame'][0] == signalType):
                signalIndex = 0
            elif(fields['signame'][1] == signalType):
                signalIndex = 1
            readData(directory + '/' + os.path.splitext(file)[0])
    print("\nTotal signals ", len(signalArray))

# Retrieve all data from files and return the x and y data
def loadAndSlice(sigType="MLII", directory="mitdb"):
    loadAllData(sigType, directory)
    sort_annotations()
    x, y = slice_peaks(signalArray, annotationArray)
    return x, y

if __name__ == '__main__':
    ps = []
    x, y = loadAndSlice()
    #z = np.random.uniform(0,1,len(x[1]))
    #plot_signal(range(len(x[1])), x[1], z)
    #plt.show()
    x_norm, y_norm, x_abnorm, y_abnorm = binarySegment(x, y)
    print(len(x_norm))
    print(len(x_abnorm))
    for i in range(len(categories)):
        ps.append(mpatches.Patch(color=colours[i], label=categories[i]))
        plotSignal([x[j] for j in range(len(x)) if y[j][i] == 1][:30], [n for n in y if n[i] == 1][:30], alpha_value=0.7)
    plt.legend(handles=ps)
    plt.show()

'''

loadAllData()

x, y = slice_peaks(signalArray, annotationArray)
x_norm, y_norm, x_abnorm, y_abnorm = binarySegment(x, y)
print(len(x_norm))
print(len(x_abnorm))
#plotSignal(x_norm[:20], y_norm[:20])
#plotSignal(x_abnorm[:20], y_abnorm[:20])
#plt.show()

#print(len(x_norm))
#print(len(x_abnorm))
#norm_mean = x_norm.mean(axis=0)
#abnorm_mean = x_abnorm[:len(x_norm)].mean(axis=0)

#norm_max = np.amin(x_norm, axis=0)
#abnorm_max = np.amin(x_abnorm, axis=0)
#plt.plot(norm_max, color="blue",alpha=1)
#plt.plot(abnorm_max, color="red",alpha=1)


#plotSignal(x_norm, y_norm)
#plotSignal(x_abnorm[:len(x_norm)], y_abnorm[:len(x_norm)])


'''
#x = np.load("test_numpy.npy")
#print(x[0], len(x))
#correctAnno()           #correct annotations
#outputArray = outputDataAs1dArray() #get data as array of 1d arrays
#outputDataAsImage()
#print(outputArray[:,0][0][0])
#writeArrayToCSV(outputArray, "100CSV")
#appendArrayToCSV(outputArray, "100CSV")
