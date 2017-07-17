import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RNN:
	sess = tf.Session()
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
	def __init__(self, data_formatter, epochs=1000, learning_rate=0.01, batch_size=128, learning_rate_reduction=1, reduction_index=1):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.data_formatter = data_formatter
		self.learning_rate_reduction = learning_rate_reduction
		self.reduction_index = reduction_index

	def setup(self):
		self.x = tf.placeholder("float", [None, self.n_step, self.n_input])
		self.y = tf.placeholder("float", [None, self.n_output])
		# Fully connected weights
		self.w1 = tf.Variable(tf.random_normal([self.n_state, 300]))
		self.b1 = tf.Variable(tf.random_normal([300]))
		self.w2 = tf.Variable(tf.random_normal([300, self.n_output]))
		self.b2 = tf.Variable(tf.random_normal([self.n_output]))
		self.output = self.feed_forward(self.x)

	def feed_forward(self, x):
		# Convert [batch, step, input] to [batch, input]
		x = tf.unstack(x, self.n_step, 1)
		# Create lstm cell
		lstm_cell = rnn.LSTMCell(self.n_state, forget_bias=1.0)
		# Run lstm_cell
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
		# Multiply output
		fc1_out = tf.matmul(outputs[-1], self.w1) + self.b1
		out = tf.matmul(fc1_out, self.w2) + self.b2
		# Return final output
		return out

	def test(self):
		# Evaluate model
		correct_pred = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		total_accuracy = 0.0

		for j in range(int(len(self.data_formatter.x_test)/self.batch_size)):
			batch_x, batch_y = self.data_formatter.get_batch(self.batch_size, j, "test")
			# Convert shape [batch * steps] -> [batch * steps * inputs]
			batch_x = np.expand_dims(batch_x, axis=2)
			# Run test over batc
			acc = self.sess.run([accuracy], feed_dict={self.x:batch_x, self.y:batch_y})
			total_accuracy = total_accuracy + acc[0]
		return total_accuracy / float(len(self.data_formatter.x_test)/self.batch_size)

	def train(self):
		# Define loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

		init = tf.global_variables_initializer()
		self.sess.run(init)
		# Begin training
		for i in range(self.epochs):
			epoch_err = 0.0
			# Reduces the learning rate based on specified index
			if(i % self.reduction_index == 0):
				self.learning_rate = self.learning_rate * self.learning_rate_reduction
			# Run training loop
			for j in range(int(len(self.data_formatter.x_train)/self.batch_size)):
				batch_x, batch_y = self.data_formatter.get_batch(self.batch_size, j, "train")
				# Convert shape [batch * steps] -> [batch * steps * inputs]
				batch_x = np.expand_dims(batch_x, axis=2)
				err, _ = self.sess.run([cost, optimizer], feed_dict={self.x:batch_x, self.y:batch_y})
				epoch_err = epoch_err + err
				sys.stdout.write("\rEpoch " + str(i + 1) + " training ... {0:.2f}%".format((float((j * self.batch_size)/len(self.data_formatter.x_train)))*100))
				sys.stdout.flush()
			sys.stdout.write("\rEpoch " + str(i + 1) + " training ... complete!")
			print("\nError:", epoch_err, "\nAccuracy:", self.test(), "\n")
