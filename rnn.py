
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
	def __init__(self, learning_rate=0.1, n_step=360, n_input=1, n_state=50, n_output=2, name="temp_name"):
		self.learning_rate = learning_rate
		self.n_step = n_step
		self.n_state = n_state
		self.n_output = n_output
		self.n_input = n_input
		self.name = name

	def setup(self, sess):
		self.sess = sess
		#Placeholders
		self.x = tf.placeholder("float", [None, self.n_step, self.n_input], name="x_place")
		self.y = tf.placeholder("float", [None, self.n_output], name="y_place")
		# Weights and biases
		self.w1 = tf.Variable(tf.truncated_normal([self.n_state, 100], stddev=0.1), name="w1")
		self.b1 = tf.constant(0.1, shape=[100], name="b1")
		self.w2 = tf.Variable(tf.truncated_normal([100, self.n_output], stddev=0.1), name="w2")
		self.b2 = tf.constant(0.1, shape=[self.n_output], name="b2")
		# Output
		self.output = self.feed_forward(self.x)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y), name="cost")
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		tf.add_to_collection("optimizer", self.optimizer)
		self.correct_pred = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1), name="correct_pred")
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
		tf.add_to_collection("correct_pred", self.correct_pred)
		tf.add_to_collection("accuracy", self.accuracy)

		sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def load(self, sess, learning_rate=0.1, name="temp_name"):
		self.name = name
		self.learning_rate = learning_rate
		self.sess = sess
		saver = tf.train.import_meta_graph('model/' + name + ".meta")
		saver.restore(sess, tf.train.latest_checkpoint('model/'))
		# Assign the graph to all variables
		graph = tf.get_default_graph()
		self.x = graph.get_tensor_by_name("x_place:0")
		self.y = graph.get_tensor_by_name("y_place:0")
		self.w1 = graph.get_tensor_by_name("w1:0")
		self.b1 = graph.get_tensor_by_name("b1:0")
		self.w2 = graph.get_tensor_by_name("w2:0")
		self.b2 = graph.get_tensor_by_name("b2:0")
		self.output = self.feed_forward(self.x)
		self.cost = graph.get_tensor_by_name("cost:0")
		self.optimizer = tf.get_collection("optimizer")[0]
		self.correct_pred = tf.get_collection("correct_pred")[0]
		self.accuracy = tf.get_collection("accuracy")[0]
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())

	def feed_forward(self, x):
		# Convert [batch, step, input] to [batch, input]
		x = tf.unstack(x, self.n_step, 1)
		#state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable([1, n_state]), tf.Variable([1, n_state]))
		# Forward direction cell
		lstm_fw_cell = rnn.BasicLSTMCell(self.n_state/2, forget_bias=1.0)
		# Backward direction cell
		lstm_bw_cell = rnn.BasicLSTMCell(self.n_state/2, forget_bias=1.0)
		# 1-layer LSTM with n_hidden units.
		#cell = rnn.LSTMCell(self.n_state, use_peepholes=True)
		# generate prediction
		outputs, _, _  = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
		# we only want the last output
		fc1_out = tf.nn.tanh(tf.matmul(outputs[-1], self.w1) + self.b1)
		output = tf.matmul(fc1_out, self.w2) + self.b2
		return output

	def test(self, features, labels):
		return self.sess.run([self.accuracy, self.correct_pred], feed_dict={self.x:features, self.y:labels})

	def train(self, features, labels):
		return self.sess.run([self.cost, self.optimizer], feed_dict={self.x:features, self.y:labels})

	def save(self, step):
		self.saver.save(self.sess, "model/" + self.name, global_step=step)

	def __call__(self, features, labels):
		return self.sess.run([self.output], feed_dict={self.x:features, self.y:labels})

if __name__ == "__main__":

	#Get Data_Formatter
	#data_formatter = setup_data()
	#data_formatter.shuffle()
	#data_formatter.noise_generator(0.1)
	#data_formatter.equalize_data()
	data_formatter = Data_Formatter()
	data_formatter.load_formatted_data("360_no_noise_binary_equalize")
	data_formatter.shuffle()
	data_formatter.split_training_testing(0.1)
	data_formatter.equalize_data()
	print(len(data_formatter.x_train))
	norm = 0
	abnorm = 0
	for y in data_formatter.y_test:
		if(y[0] == 1):
			norm = norm + 1
		elif(y[1] == 1):
			abnorm = abnorm + 1
		else:
			print(y)
	print(norm)
	print(abnorm)
	#Hyperparameters
	epochs = 100
	batch_size = int(len(data_formatter.x_test)/10)


	rnn_network = RNN(n_input=1, n_state=100, n_step=360, n_output=2, name="binary_classification")
	with tf.Session() as sess:
		#rnn_network.setup(sess)
		rnn_network.load(sess, name="binary_classification-0", learning_rate=0.001)
		for i in range(epochs):
		    epoch_err = 0.0
		    # Run training loop
		    for j in range(int(len(data_formatter.x_train)/batch_size)):
		        batch_x, batch_y = data_formatter.get_batch(batch_size, j, "train")
		        # Convert shape [batch * steps] -> [batch * steps * inputs]
		        batch_x = np.expand_dims(batch_x, axis=2)
		        err, _ = rnn_network.train(batch_x, batch_y)
		        epoch_err = epoch_err + err
		        sys.stdout.write("\rEpoch " + str(i + 1) + " training ... {0:.2f}%".format((float((j * batch_size)/len(data_formatter.x_train)))*100))
		        sys.stdout.flush()
		        prev_epoch_err = epoch_err
		    #rnn_network.learning_rate = rnn_network.learning_rate * 0.9
		    rnn_network.save(0)
		    sys.stdout.write("\rEpoch " + str(i + 1) + " training ... complete!")

		    total_accuracy = 0.0
		    output_errs = np.zeros(2)
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
