#BASIC NN with Tensorboard - Aidan
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from data_formatter import Data_Formatter
import random
now = datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Importing Data")
signals, labels = [], []
signals = np.load("data/features_MLII.npy")
labels = np.load("data/labels_MLII.npy")
'''
def checkLabels():
    countNorm = 0
    countAb = 0
    newLabels = []
    newSignals = []
    for i in range(0, len(label)):
        if(label[i][0] == 1.0):
            countNorm += 1
        else:
            countAb += 1
            newLabels.append(label[i])
            newSignals.append(signal[i])
    print("Normal: ", countNorm)
    print("Abnormal: ", countAb)
    cN = 0
    for i in range(0, len(label)):
        if(label[i][0] == 1.0 and cN < countAb):
            newLabels.append(label[i])
            newSignals.append(signal[i])
            cN += 1
    return newLabels, newSignals

labels, signals = checkLabels()
'''

def createBatch(signals, labels, noBatch):
    allSignals = np.array_split(signals, noBatch)
    allLabels = np.array_split(labels, noBatch)
    return allSignals, allLabels

def splitTestTrainSets(testPercentage):
    testNo = int(100 - testPercentage)
    trainNo = 100 - testNo

    #spliting array evenly by 100
    splitSignalArr = np.array_split(signals, 100)
    splitLabelArr = np.array_split(labels, 100)

    #Seperating training and testing set
    signalTrainSet = np.concatenate(splitSignalArr[:trainNo])
    labelTrainSet = np.concatenate(splitLabelArr[:trainNo])
    signalTestSet = np.concatenate(splitSignalArr[trainNo:])
    labelTestSet = np.concatenate(splitLabelArr[trainNo:])
    print(len(signalTestSet))
    print(len(signalTrainSet))
    return signalTrainSet, labelTrainSet, signalTestSet, labelTestSet

with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, 360]) #Creates an input placeholder with input of 360 (signal length as 1d Array)
    y_ = tf.placeholder(tf.float32, [None, 2]) #Placeholder for inputing correct answers

with tf.name_scope("Layer1"):
    W = tf.Variable(tf.random_normal([360, 2], stddev=0.1))
    tf.summary.histogram("weights", W)
    b = tf.Variable(tf.zeros([2]))

#y1 = tf.nn.softmax(tf.matmul(x, h1w) + h1b)
#y = tf.nn.softmax(tf.matmul(y1, W) + b)

with tf.name_scope("output"):
    y = tf.matmul(x, W) + b

#TRAINING
with tf.name_scope("training"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#Evaluating Model
with tf.name_scope("evaluation"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

#Run Session
df = Data_Formatter()

print("ct", df.countType(labels))
df.assign_data(signals, labels)

print("S, L: ", len(labels), len(signals))
print("ct", df.countType(df.y))
df.equalize_data()
print("ct", df.countType(df.y))
df.split_training_testing()
print("ytest", df.countType(df.y_test))
print("ytrain", df.countType(df.y_train))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
#logdir = "tensorboard" + now.strftime("%Y%m%d-%H%M%S") + "/"


summary, acc = sess.run([merged, accuracy], feed_dict={x: df.x_test, y_: df.y_test})
print("ACCURACY", acc)
for i in range(0, 10):
    sess.run(train_step, feed_dict={x: df.x_train, y_: df.y_train})
    summary, acc, ce = sess.run([merged, accuracy, cross_entropy], feed_dict={x: df.x_test, y_: df.y_test})
    print("accuracy: ", acc, " ce:", ce)
#Main
'''
train_writer = tf.summary.FileWriter('tensorboard' + '/' + now.strftime("%Y%m%d-%H%M%S"), sess.graph)
signalTrainSet, labelTrainSet, signalTestSet, labelTestSet = splitTestTrainSets(5)
signalTrainSet, labelTrainSet = createBatch(signalTrainSet, labelTrainSet, 20)
summary, acc = sess.run([merged, accuracy], feed_dict={x: signalTestSet, y_: labelTestSet})
print("ACCURACY", acc)

for i in range(0, len(signalTrainSet)):
    print("Training Batch ", i, "of ", len(signalTrainSet))
    sess.run(train_step, feed_dict={x: signalTrainSet[i], y_: labelTrainSet[i]})
    summary, acc = sess.run([merged, accuracy], feed_dict={x: signalTestSet, y_: labelTestSet})
    print("Batch ", i, " accuracy: ", acc)
    train_writer.add_summary(summary, i)
train_writer.close()
    '''
