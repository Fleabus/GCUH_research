
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RNN:
	n_input = 1
	n_step = 360
	n_state = 124
	n_output = 2


	'''
	@params
	class   ~ data_formatter
	int     ~ epochs
	float   ~ learning_rate
	int     ~ batch_size
	'''
	def __init__(self, learning_rate=0.01, n_step=360, n_input=1, n_state=124, n_output=2):
		self.learning_rate = learning_rate
		self.n_step = n_step
		self.n_state = n_state
		self.n_output = n_output
		self.n_input = n_input

	def setup(self, sess):
		self.sess = sess
		self.w1 = tf.Variable(tf.truncated_normal([self.n_state, 2], stddev=0.1))
		self.b1 = tf.constant(0.1, shape=[2])
		self.x = tf.placeholder("float", [None, self.n_step, self.n_input])
		self.y = tf.placeholder("float", [None, self.n_output])
		self.output = self.feed_forward(self.x)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.correct_pred = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		sess.run(tf.global_variables_initializer())

	def feed_forward(self, x):
		# Convert [batch, step, input] to [batch, input]
		x = tf.unstack(x, self.n_step, 1)


		'''
		# Create lstm cell(s)
		fwd_lstm_cell = rnn.LSTMCell(self.n_state/2)
		bwd_lstm_cell = rnn.LSTMCell(self.n_state/2)
		# Run lstm_cell
		try:
			outputs, _, _ = rnn.static_bidirectional_rnn(fwd_lstm_cell, bwd_lstm_cell, x, dtype=tf.float32)
		except Exception:
			outputs = rnn.static_bidirectional_rnn(fwd_lstm_cell, bwd_lstm_cell, x, dtype=tf.float32)
		# Multiply outputb

		fc1_result = tf.matmul(outputs[-1], self.w1) + self.b1
		fc1_out = tf.nn.sigmoid(fc1_result)
		fc2_result = tf.matmul(fc1_out, self.w2) + self.b2
		out = tf.nn.sigmoid(fc2_result)
		'''
		lstm_cell = rnn.BasicLSTMCell(self.n_state, state_is_tuple=True)
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
		output = tf.matmul(outputs[-1],self.w1) + self.b1
		# Return final output
		return output

	def test(self, features, labels):
		return self.sess.run([self.accuracy, self.correct_pred], feed_dict={self.x:features, self.y:labels})

	def train(self, features, labels):
		return self.sess.run([self.cost, self.optimizer], feed_dict={self.x:features, self.y:labels})
