
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from setup import setup_data
from data_formatter import Data_Formatter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RNN:
	n_input = 1
	n_step = 360
	n_state = 400
	n_output = 2

	'''
	@params
	class   ~ data_formatter
	int     ~ epochs
	float   ~ learning_rate
	int     ~ batch_size
	'''
	def __init__(self, learning_rate=1, n_step=360, n_input=1, n_state=120, n_output=2):
		self.learning_rate = learning_rate
		self.n_step = n_step
		self.n_state = n_state
		self.n_output = n_output
		self.n_input = n_input

	def setup(self, sess):
		self.sess = sess
		#Placeholders
		self.x = tf.placeholder("float", [None, self.n_step, self.n_input])
		self.y = tf.placeholder("float", [None, self.n_output])
		# Weights and biases
		self.w1 = tf.Variable(tf.truncated_normal([self.n_state, 50], stddev=0.1))
		self.b1 = tf.constant(0.1, shape=[50])
		self.w2 = tf.Variable(tf.truncated_normal([50, self.n_output], stddev=0.1))
		self.b2 = tf.constant(0.1, shape=[self.n_output])
		# Output
		self.output = self.feed_forward(self.x)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.correct_pred = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		sess.run(tf.global_variables_initializer())

	def feed_forward(self, x):
		# Convert [batch, step, input] to [batch, input]
		x = tf.unstack(x, self.n_step, 1)
		#state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable([1, n_state]), tf.Variable([1, n_state]))
		# 1-layer LSTM with n_hidden units.
		cell = rnn.LSTMCell(self.n_state, use_peepholes=True)
		# generate prediction
		outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
		# we only want the last output
		fc1_out = tf.nn.tanh(tf.matmul(outputs[-1], self.w1) + self.b1)
		output = tf.matmul(fc1_out, self.w2) + self.b2
		return output

	def test(self, features, labels):
		return self.sess.run([self.accuracy, self.correct_pred], feed_dict={self.x:features, self.y:labels})

	def train(self, features, labels):
		return self.sess.run([self.cost, self.optimizer], feed_dict={self.x:features, self.y:labels})

	def __call__(self, features, labels):
		return self.sess.run([self.output], feed_dict={self.x:features, self.y:labels})

if __name__ == "__main__":

	#Get Data_Formatter
	#data_formatter = setup_data()
	#data_formatter.shuffle()
	#data_formatter.noise_generator(0.1)
	#data_formatter.equalize_data()
	data_formatter = Data_Formatter()
	data_formatter.load_formatted_data("360_no_noise")
	data_formatter.split_training_testing(0.01)

	#Hyperparameters
	epochs = 100
	batch_size = int(len(data_formatter.x_test)/20)


	rnn_network = RNN(n_input=5, n_step=int(360/5), n_output=7)

	with tf.Session() as sess:
		rnn_network.setup(sess)
		for i in range(epochs):
		    epoch_err = 0.0
		    # Run training loop
		    for j in range(int(len(data_formatter.x_train)/batch_size)):
		        batch_x, batch_y = data_formatter.get_batch(batch_size, j, "train")
		        # Convert shape [batch * steps] -> [batch * steps * inputs]
		        #batch_x = np.resize(batch_x, [batch_size, int(len(batch_x[0])/5), 5])
		        batch_x = np.expand_dims(batch_x, axis=2)
		        err, _ = rnn_network.train(batch_x, batch_y)
		        epoch_err = epoch_err + err
		        sys.stdout.write("\rEpoch " + str(i + 1) + " training ... {0:.2f}%".format((float((j * batch_size)/len(data_formatter.x_train)))*100))
		        sys.stdout.flush()
		        prev_epoch_err = epoch_err
		    #rnn_network.learning_rate = rnn_network.learning_rate * 0.9
		    sys.stdout.write("\rEpoch " + str(i + 1) + " training ... complete!")

		    total_accuracy = 0.0
		    output_errs = np.zeros(7)
		    for j in range(int(len(data_formatter.x_test)/batch_size)):
		        batch_x, batch_y = data_formatter.get_batch(batch_size, j, "test")
		        # Convert shape [batch * steps] -> [batch * steps * inputs]
		        #batch_x = np.resize(batch_x, [batch_size, int(len(batch_x[0])/5), 5])
		        batch_x = np.expand_dims(batch_x, axis=2)
		        # Run test over batc
		        acc, corr = rnn_network.test(batch_x, batch_y)
		        output_errs = np.add(output_errs, np.sum([batch_y[x] for x in range(len(corr)) if corr[x] == False], axis=0))
		        total_accuracy = total_accuracy + acc
		    print("\nError:", epoch_err, "\nAccuracy:", total_accuracy / (len(data_formatter.x_test)/batch_size), "\nCategory Errors:", output_errs, "\n")
