from setup import setup_data
import numpy as np
from rnn import RNN
from data_formatter import Data_Formatter
import tensorflow as tf
import sys
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

# Create rnn
n_input = 1
n_step = 360
n_state = 50
n_output = 2
learning_rate = 0.5
#rnn = RNN(learning_rate=0.00001)
w1 = tf.Variable(tf.truncated_normal([n_state, 100], stddev=0.1))
b1 = tf.constant(0.1, shape=[100])
w2 = tf.Variable(tf.truncated_normal([n_state, 2], stddev=0.1))
b2 = tf.constant(0.1, shape=[2])

x = tf.placeholder("float", [None, n_step, n_input])
y = tf.placeholder("float", [None, n_output])


def feed_forward(x):
    # Convert [batch, step, input] to [batch, input]
    x = tf.unstack(x, n_step, 1)
    #state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable([1, n_state]), tf.Variable([1, n_state]))
    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.LSTMCell(n_state, use_peepholes=True)


    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # we only want the last output
    return tf.matmul(outputs[-1], w2) + b2

if __name__ == "__main__":

    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())

    data_formatter.shuffle()
    data_formatter.equalize_data()
    '''
    x_temp = []
    y_temp = []
    for i in range(10000):
        if(i % 2 == 0):
            x_temp.append(np.zeros((360)))
            y_temp.append([1, 0])
        else:
            x_temp.append(np.ones((360)))
            y_temp.append([0, 1])

    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)

    data_formatter = Data_Formatter()
    data_formatter.assign_data(x_temp, y_temp)
    '''
    #temp = np.array(data_formatter.average_in_window(60))
    #print(temp.shape)
    data_formatter.split_training_testing(0.1)

    #Hyperparameters
    epochs = 100
    batch_size = int(len(data_formatter.x_test)/10)

    output = feed_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        #rnn.setup(sess)
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            epoch_err = 0.0
            # Run training loop
            for j in range(int(len(data_formatter.x_train)/batch_size)):
                batch_x, batch_y = data_formatter.get_batch(batch_size, j, "train")
                # Convert shape [batch * steps] -> [batch * steps * inputs]
                batch_x = np.expand_dims(batch_x, axis=2)
                err, _ = sess.run([cost, optimizer], feed_dict={x:batch_x, y:batch_y})
                epoch_err = epoch_err + err
                sys.stdout.write("\rEpoch " + str(i + 1) + " training ... {0:.2f}%".format((float((j * batch_size)/len(data_formatter.x_train)))*100))
                sys.stdout.flush()
                prev_epoch_err = epoch_err
            sys.stdout.write("\rEpoch " + str(i + 1) + " training ... complete!")

            acc_x = []
            acc_y = []
            total_accuracy = 0.0
            for j in range(int(len(data_formatter.x_test)/batch_size)):
                batch_x, batch_y = data_formatter.get_batch(batch_size, j, "test")
                # Convert shape [batch * steps] -> [batch * steps * inputs]
                batch_x = np.expand_dims(batch_x, axis=2)
                # Run test over batc
                acc, corr = sess.run([accuracy, correct_pred], feed_dict={x:batch_x, y:batch_y})
                acc_x.extend(batch_x)
                acc_y.extend(corr)
                total_accuracy = total_accuracy + acc
            data_formatter.plot_accuracy(acc_x, acc_y)
            plt.show()
            print("\nError:", epoch_err, "\nAccuracy:", total_accuracy / (len(data_formatter.x_test)/batch_size) , "\n")
