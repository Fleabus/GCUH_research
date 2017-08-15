#File to detect RR Interval in ascii data and output to image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
from tqdm import *

def plotGraph(signalArray, indexArray, patientNumber, quantity = 99999, width=5.5, height=3, dpi=150):
    print("Plotting and Saving Snapshots...")
    if quantity > len(signalArray):
        for i in tqdm(range(0, len(signalArray))):
            plt.figure(figsize=(width,height), dpi=dpi, frameon='false', tight_layout={'pad': 0})
            ax = plt.gca()
            plt.minorticks_on()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(12))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
            ax.grid(color='r', which='major', linestyle='-', linewidth=.3)
            ax.grid(color='b', which='minor', linestyle='--', linewidth=.1)
            plt.plot(signalArray[i], linewidth=2.5)
            #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            plt.savefig("allECGs/%s-%s.png" % (patientNumber, indexArray[i]), bbox_inches='tight', pad_inches = 0)
            plt.close()
    else:
        for i in tqdm(range(0, quantity)):
            plt.figure(figsize=(width,height), dpi=dpi, frameon='false', tight_layout={'pad': 0})
            ax = plt.gca()
            plt.minorticks_on()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(12))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
            ax.grid(color='r', which='major', linestyle='-', linewidth=.3)
            ax.grid(color='b', which='minor', linestyle='--', linewidth=.1)
            plt.plot(signalArray[i], linewidth=2.5)
            #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            plt.savefig("allECGs/%s-%s.png" % (patientNumber, indexArray[i]), bbox_inches='tight', pad_inches = 0)
            plt.close()
    return

def chopSignal(allSignals, leeway):
    signalStart = 0
    bufferPercentage = 10
    signalStart = detectStart(allSignals)
    rWaveValues = getRPeaks(allSignals, signalStart)

    rAverage = sum(rWaveValues)/float(len(rWaveValues)) #works out avg of initial round of r peak detection
    rWavePeakIndex = findIndexOfValues(allSignals, rAverage-leeway)
    rWaveIndexes = correctDetection(allSignals, rWavePeakIndex, 200)
    return rWaveIndexes

#correctRClassification
def correctDetection(signalArray, indexArray, checkRange):
    tempArray = []
    print("Validating R Waves...")
    for i in tqdm(range(0, (len(indexArray)-2))):
        if ((indexArray[i+1] - indexArray[i]) < checkRange):
            if (signalArray[indexArray[i]] < signalArray[indexArray[i+1]]):
                tempArray.append(i)
            else:
                tempArray.append(i+1)

    for i in range(len(tempArray), 0, -1):
        indexArray.pop(tempArray[i-1])
    return indexArray

def checkMax(signalArray):
    for i in range(0, len(signalArray)):
        max_value = max(signalArray[i])

#detect where signal begins
def detectStart(array):
    for i in range(2, len(array)):
        if (array[i] != '0.00'):
            signalStart = i
            return signalStart

#form a list of all R Wave peaks
def getRPeaks(signalArray, signalStart):
    rWaveValues = []
    print("Calculating average R Wave Peak...")
    for i in tqdm(range(0, int((len(signalArray)-signalStart)/300))):
        currentBuffer = signalStart+(i*300)
        tempList = signalArray[currentBuffer:currentBuffer+300]
        max_value = max(tempList)
        rWaveValues.append(float(max_value))
    return rWaveValues

def findIndexOfValues(array, value):
    indexArray = []
    print("Finding all R Waves within given range...")
    for i in tqdm(range(2, len(array)-1)):
        if float(array[i]) > value and float(array[i]) > float(array[i-1]) and float(array[i]) > float(array[i+1]):
            indexArray.append(i)
    return indexArray

def snapshotArray(indexArray, signalArray, length = 150):
    correctArray = []
    print("Taking snapshot of each QRS Complex")
    for i in tqdm(range(1, len(indexArray)-1)): #shortcut to ensure no indexOutBounds
        tempArray = signalArray[indexArray[i]-length:indexArray[i]+length]
        correctArray.append(tempArray)
    return correctArray

def readInAscii(filename):
    signalArray = []
    print("Reading in: ", filename)
    with open(filename) as f:
        for line in f:
            signalArray.append(line.rstrip('\n'))
    return signalArray

def outputMultiplePatientImages(filename, leniency, noEach = 9999999):
    allSignalArray = readInAscii(filename)
    rWaveIndexes = chopSignal(allSignalArray, leniency)
    patientNumber = allSignalArray[0].split()[-1]
    plotGraph(snapshotArray(rWaveIndexes, allSignalArray, 500), rWaveIndexes, patientNumber, quantity=noEach, width=16, height=8, dpi=150)

#outputMultiplePatientImages('SAH/testData.asc', 0.35, 10)
#outputMultiplePatientImages('SAH/2757.asc', 0.35) done 10141 of these
outputMultiplePatientImages('SAH/2987_1.asc', 0.35, 1000)

'''
outputMultiplePatientImages('SAH/testData.asc', 0.35, 10)
outputMultiplePatientImages('SAH/2757.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_1.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_2.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_3.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_4.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_5.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/2987_6.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/3022_1.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/3022_2.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/3022_3.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/3022_4.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/3339_1.asc', 0.35, 1000)
outputMultiplePatientImages('SAH/4877_1.asc', 0.35, 1000)
'''
