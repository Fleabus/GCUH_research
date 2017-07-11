import wfdb
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import sys
import os

#Categories 17
categories = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f']
colours = ['#e6194b	', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6'
            ,'#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000']

#HyperParameters
signalType = "V1"
hz = 360
SecondsWanted = 2500 #Seconds of data wanted
dynamicPeak = False # If true, the ecg slice will pick based on if the highest or lowest point is greater
peakSelection = 1 # 1 == pick highest point. -1 == pick lowest point. This only works if dynamic peak is false
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
        if(annotationArray[i-1][1] == 'N' or annotationArray[i-1][1] == '.'):
            #print(annotation.anntype[i])
            annotationArray[i-1][1] = 0
        elif(annotationArray[i-1][1] == 'A'):
            #print(annotation.anntype[i])
            annotationArray[i-1][1] = 1
        else:
            x = 1
            #print("Unknown annotation ", annotationArray[i-1][1])
            #annotationArray[i-1][1] = -1



'''
Signals two-tuple [[sig1, sig2], [sig1, sig2] ... [sig1, sig2]]
Annoations two-tuple [[index, type], [index, type] ... [index, type]]

Find highest peak of signal within 300hz of annotation index
Slice peak 300hz either way
'''
def slice_peaks(signals, annotations):
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
            maxIndex = np.argmax(signalRange)
            minIndex = np.argmin(signalRange)
            average = np.mean(signalRange)
            if(abs(signalRange[maxIndex] - average) < abs(signalRange[minIndex] - average)):
                peakIndex = minIndex
            else:
                peakIndex = maxIndex
            if(dynamicPeak == False):
                if(peakSelection == 1):
                    peakIndex = maxIndex
                if(peakSelection == -1):
                    peakIndex = minIndex

            peakIndex += low
            high = int(peakIndex + hz/2)
            low = int(peakIndex - hz/2)
            if(high > len(signals)):
                assign = False
            if(low < 0):
                assign = False
            if assign:
                signalRange = signals[low:high]
                x.append([i[signalIndex] for i in signalRange])
                if(label == 0):
                    tempLabel = [1, 0] # [normal, abnormal]
                elif(label == 1):
                    tempLabel = [0, 1] # [normal, abnormal]
                else:
                    tempLabel = [0, 0]
                y.append(tempLabel)

        # Assign lists to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y


def plotSignal(features, labels):
    plt.style.use('dark_background')
    for i in range(len(features)):
        if(labels[i][0] == 1):
            plt.plot(features[i], color="blue",alpha=0.01)
        else:
            plt.plot(features[i], color="red", alpha=0.01)
        #else:
            #plt.plot(features[i], color="yellow", alpha=0.3)


# returns the two arrays
def retrieveData():
    return signalArray, annotationArray


def binarySegment(x, y):
    x_norm = []
    y_norm = []
    x_abnorm = []
    y_abnorm = []

    for i in range(len(x)):
        x[i] = (x[i] - x[i].min(0)) / x[i].ptp(0)
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
    x, y = slice_peaks(signalArray, annotationArray)
    return x, y

'''
loadAllData()

x, y = slice_peaks(signalArray, annotationArray)
x_norm, y_norm, x_abnorm, y_abnorm = binarySegment(x, y)
#print(len(x_norm))
#print(len(x_abnorm))
#norm_mean = x_norm.mean(axis=0)
#abnorm_mean = x_abnorm[:len(x_norm)].mean(axis=0)

norm_max = np.amin(x_norm, axis=0)
abnorm_max = np.amin(x_abnorm, axis=0)
plt.plot(norm_max, color="blue",alpha=1)
plt.plot(abnorm_max, color="red",alpha=1)

#plotSignal(x_norm, y_norm)
#plotSignal(x_abnorm[:len(x_norm)], y_abnorm[:len(x_norm)])
plt.show()
'''

'''
#x = np.load("test_numpy.npy")
#print(x[0], len(x))
#correctAnno()           #correct annotations
#outputArray = outputDataAs1dArray() #get data as array of 1d arrays
#outputDataAsImage()
#print(outputArray[:,0][0][0])
#writeArrayToCSV(outputArray, "100CSV")
#appendArrayToCSV(outputArray, "100CSV")
'''
