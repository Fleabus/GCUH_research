import wfdb
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import sys

#HyperParameters
SecondsWanted = 600 #Seconds of data wanted

#Global Variables
hz = 360
signalIndex = 1
sampSize = int(SecondsWanted*360) #Sets sample size to number of seconds by Hz(360)
sampIncrement = 1/360
annotationArray = []
signalArray = []


#Initially read data in
def readData(filename):
    global annotationArray
    global signalArray
    #read in data using
    '''    record = wfdb.rdsamp(filename, sampto = sampSize)
    annotation = wfdb.rdann(filename, 'atr', sampto = sampSize)
    sig, fields = wfdb.srdsamp(filename, sampto = sampSize)'''
    record = wfdb.rdsamp(filename)
    annotation = wfdb.rdann(filename, 'atr')
    sig, fields = wfdb.srdsamp(filename)
    print(annotation.annsamp[0])
    print(annotation.anntype[0])
    #read record and signal into an array
    print("Reading in ", len(sig)/300, " seconds of data")
    print(fields['signame'])

    for i in range(0, len(sig)):
        if(i%100 == 0):
            sys.stdout.write("\rReading Data ... {0:.2f}%".format((float(i)/len(sig))*100))
            sys.stdout.flush()

        signalArray.append(sig[i])

    for i in range(1, len(annotation.annsamp)):
        annotationArray.append([annotation.annsamp[i], annotation.anntype[i]])
        if(annotationArray[i-1][1] == 'N' or annotationArray[i-1][1] == '.'):
            annotationArray[i-1][1] = 0
        elif(annotationArray[i-1][1] == 'A'):
            annotationArray[i-1][1] = 1
        else:
            print("Unknown annotation ", annotationArray[i-1][1])
            annotationArray[i-1][1] = -1

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
                    tempLabel = [0, 1] # [normal, abnormal0]
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
        elif(labels[i][1] == 1):
            plt.plot(features[i], color="red", alpha=0.5)
        else:
            plt.plot(features[i], color="yellow", alpha=0.3)



readData('mitdb/100')   #read data in
x_norm = []
y_norm = []

x_abnorm = []
y_abnorm = []

x, y = slice_peaks(signalArray, annotationArray)
#x = (x - x.min(0)) / x.ptp(0)

for i in range(len(x)):
    x[i] = (x[i] - x[i].min(0)) / x[i].ptp(0)
    if(y[i][0] == 1):
        x_norm.append(x[i])
        y_norm.append(y[i])
    elif(y[i][1] == 1):
        x_abnorm.append(x[i])
        y_abnorm.append(y[i])

x_norm = np.array(x_norm)
y_norm = np.array(y_norm)
x_abnorm = np.array(x_abnorm)
y_abnorm = np.array(y_abnorm)


plotSignal(x_norm, y_norm)
plotSignal(x_abnorm, y_abnorm)
plt.show()
#x = np.load("test_numpy.npy")
#print(x[0], len(x))
#correctAnno()           #correct annotations
#outputArray = outputDataAs1dArray() #get data as array of 1d arrays
#outputDataAsImage()
#print(outputArray[:,0][0][0])
#writeArrayToCSV(outputArray, "100CSV")
#appendArrayToCSV(outputArray, "100CSV")
