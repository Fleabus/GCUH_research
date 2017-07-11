import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class RNN:
    n_input = 1
    n_step = 360
    n_state = 256
    n_output = 2

    '''
    @params
    class   ~ data_formatter
    int     ~ epochs
    float   ~ learning_rate
    int     ~ batch_size
    '''
    def __init__(self, data_formatter, epochs=1000, learning_rate=0.01, batch_size=128):
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.data_formatter = data_formatter

    def setup(self):
        self.x = tf.placeholder("float", [None, self.n_step, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_output])
        self.w = tf.Variable(tf.random_normal([self.n_state, self.n_output]))
        self.b = tf.Variable(tf.random_normal([self.n_output]))
        self.output = self.feed_forward(self.x)

    def feed_forward(self, x):
        # Convert [batch, step, input] to [batch, input]
        x = tf.unstack(x, self.n_step, 1)
        # Create lstm cell
        lstm_cell = rnn.BasicLSTMCell(self.n_state, forget_bias=1.0)
        # Run lstm_cell
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # Multiply output
        out = tf.matmul(outputs[-1], self.w) + self.b
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
            # Run test over batch
            acc = sess.run([accuracy], feed_dict={self.x:batch_x, self.y:batch_y})
            total_accuracy = total_accuracy + acc
        return total_accuracy / int(len(self.data_formatter.x_test)/self.batch_size)
        
    def train(self):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # Begin training
            for i in range(self.epochs):
                epoch_err = 0.0
                for j in range(int(len(self.data_formatter.x_train)/self.batch_size)):
                    batch_x, batch_y = self.data_formatter.get_batch(self.batch_size, j, "train")
                    # Convert shape [batch * steps] -> [batch * steps * inputs]
                    batch_x = np.expand_dims(batch_x, axis=2)
                    err, _ = sess.run([cost, optimizer], feed_dict={self.x:batch_x, self.y:batch_y})
                    epoch_err = epoch_err + err
                print("\nEpoch", i, "\nError:", epoch_err, "\nAccuracy:", self.test())
