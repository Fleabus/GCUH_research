#BASIC CNN - Aidan
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from data_formatter import Data_Formatter
now = datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Importing Data")
signals, labels = [], []
signals = np.load("data/features_MLII.npy")
labels = np.load("data/labels_MLII.npy")
'''
def checkLabels(label, signal):
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
    print("S, L: ", len(newSignals), len(newLabels))
    return newLabels, newSignals

def countType(arr):
    countNorm = 0
    countAb = 0
    for i in range(0, len(arr)):
        if(arr[i][0] == 1.0):
            countNorm += 1
        else:
            countAb += 1
    return countNorm, countAb
'''
def getSome(number):
    la = []
    sa = []
    la = label[0:number]
    sa = signal[0:number]
    return la, sa

#labels, signals = checkLabels(labels, signals)
#labels, signals = getSome(5000)
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
    return signalTrainSet, labelTrainSet, signalTestSet, labelTestSet

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, 360]) #Creates an input placeholder with input of 360 (signal length as 1d Array)
    y_ = tf.placeholder(tf.float32, [None, 2]) #Placeholder for inputing correct answers

    x_image = tf.reshape(x, [-1, 1, 360, 1])

with tf.name_scope("ConvLayer1"):
    W_conv1 = weight_variable([1, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)


with tf.name_scope("FCLayer1"):
    W_fc1 = weight_variable([1 * 180 * 32, 1024])
    b_fc1 = bias_variable([1024])
    h_pool1_flat = tf.reshape(h_pool1, [-1, 1*180*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

with tf.name_scope("output"):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

y = tf.matmul(h_fc1, W_fc2) + b_fc2

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
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
#logdir = "tensorboard" + now.strftime("%Y%m%d-%H%M%S") + "/"

train_writer = tf.summary.FileWriter('tensorboard' + '/' + now.strftime("%Y%m%d-%H%M%S"), sess.graph)

#Main
summary, acc = sess.run([merged, accuracy], feed_dict={x: df.x_test, y_: df.y_test})
print("ACCURACY", acc)
for i in range(0, 10):
    sess.run(train_step, feed_dict={x: df.x_train, y_: df.y_train})
    summary, acc = sess.run([merged, accuracy], feed_dict={x: df.x_test, y_: df.y_test})
    print("accuracy: ", acc)
train_writer.add_summary(summary)
train_writer.close()
'''
#Main
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
