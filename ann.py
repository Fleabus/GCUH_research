import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from setup import setup_data
from data_formatter import Data_Formatter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#============================================================#
#                                                            #
#   Artificial Neural Network                                #
#   Using the adult data to predict income                   #
#   - Matthew Lee-Mattner                                    #
#                                                            #
#============================================================#
# import data
data_formatter = Data_Formatter()
data_formatter.load_formatted_data("900_no_noise_binary_equalize")
#data_formatter.shuffle()
#data_formatter.normalize(-1.5, 1.5)
#data_formatter.center_vertical()
data_formatter.split_training_testing(0.3)
print(len(data_formatter.x_test))
print(len(data_formatter.x_train))


#constants
features = 900
hl_1 = 100
hl_2 = 100
hl_3 = 100
output_nodes = 2
epochs = 50000
batch_size = 100

#hyperparameters
lr = 1

# placholders
x = tf.placeholder('float', [None, features])
y = tf.placeholder('float', [None, output_nodes])

# return an object with weights and biases
def layer_setup(inputs, outputs):
    layer = {
        'weights': tf.Variable(tf.truncated_normal([inputs, outputs], stddev=0.1)),
        'biases': tf.constant(0.1, shape=[outputs])
    }
    return layer

def network_setup(x):
    # setup each layer
    hidden_layer_1 = layer_setup(features, hl_1)
    hidden_layer_2 = layer_setup(hl_1, hl_2)
    hidden_layer_3 = layer_setup(hl_2, hl_3)
    output = layer_setup(hl_3, output_nodes)
    # forward prop
    hl_1_result = tf.matmul(x, hidden_layer_1['weights']) + hidden_layer_1['biases']
    hl_1_result = tf.nn.sigmoid(hl_1_result)
    hl_2_result = tf.matmul(hl_1_result, hidden_layer_2['weights']) + hidden_layer_2['biases']
    hl_2_result = tf.nn.sigmoid(hl_2_result)
    hl_3_result = tf.matmul(hl_2_result, hidden_layer_3['weights']) + hidden_layer_3['biases']
    hl_3_result = tf.nn.sigmoid(hl_3_result)
    result = tf.matmul(hl_3_result, output['weights']) + output['biases']
    #result = tf.nn.sigmoid(result) # reduce to value between 0 and 1
    return result

def train_network(x):
    prediction = network_setup(x)
    with tf.name_scope("Optimization"):
        cost = tf.reduce_mean( tf.squared_difference(y, prediction))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(len(data_formatter.x_train)/batch_size)):
                epoch_x, epoch_y = data_formatter.get_batch(batch_size, i, "train")
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
            #Test epoch
            # Compare the predicted outcome against the expected outcome

            correct = tf.equal(tf.round(prediction), y)
            # Use the comparison to generate the accuracy
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            final_accuracy = 0
            if epoch % 100 == 0:
                for i in range(int(len(data_formatter.x_test)/batch_size)):
                    epoch_x, epoch_y = data_formatter.get_batch(batch_size, i, "test") # Magically gets the next batch
                    final_accuracy += accuracy.eval(feed_dict={x: epoch_x, y: epoch_y})
                print("test accuracy %", final_accuracy / int(len(data_formatter.x_test)/batch_size) * 100)
                final_accuracy = 0
                for i in range(int(len(data_formatter.x_train)/batch_size)):
                    epoch_x, epoch_y = data_formatter.get_batch(batch_size, i, "train") # Magically gets the next batch
                    final_accuracy += accuracy.eval(feed_dict={x: epoch_x, y: epoch_y})
                print("train accuracy %", final_accuracy / int(len(data_formatter.x_train)/batch_size) * 100)

train_network(x)
