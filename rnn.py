import numpy as np
import tensorflow as tf

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
        self.data_formatter

    def setup(self):
        self.x = tf.placeholder("float", [None, n_step, n_input])
        self.y = tf.placeholder("float", [None, n_output])
        self.w = tf.Variable(tf.random_normal([n_state, n_output]))
        self.b = tf.Variable(tf.random_normal([n_output]))
        self.output = self.feed_forward(self.x)

    def feed_forward(self, x):
        # Convert [batch, step, input] to [batch, input]
        x = tf.unstack(x, n_step, 1)
        # Create lstm cell
        lstm_cell = rnn.BasicLSTMCell(n_state, forget_bias=1.0)
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

    def train(self):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session as sess:
            sess.run(init)
            # Begin training
