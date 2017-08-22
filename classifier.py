from plot_signal import plot_signal
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide='ignore', invalid='ignore')

class Classifier:
    data = []

    def __init__(self):
        self.sess = tf.Session()

    '''
    Loads data and stores in data
    Requires all data to be in the folder 'unlabelled/'
    '''
    def load_data(self, filename):
        with open("unlabelled/"+filename) as f:
            content = f.readlines()
        content.pop(0)
        content.pop(0)
        self.data = [float(x.replace('\\n', '').strip()) for x in content]

    '''
    Loads model and assigns x_placeholder, run, and input variables
    Requires the model to be in the folder 'model/'
    '''
    def load_model(self, modelname, input_amount=900):
        loader = tf.train.import_meta_graph('model/'+ modelname +'.meta')
        loader.restore(self.sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        self.input = input_amount
        self.x_placeholder = graph.get_tensor_by_name("x:0")
        self.run = graph.get_tensor_by_name("run:0")

    '''
    Runs the provided model against the provided data
    Assigns the x, y, and z variables for graphing
    limit: The length of data you want to run the model on. If left as '0', the length of the data will be assigned
    offset: provides on offset for the limit
    threshold_min/threshold_max: These values determine how sensitive the network is in classifying the data (higher = less sensitive)
    '''
    def run_model(self, limit=0, offset=0, threshold_min=-0.7, threshold_max=0.7):
        if(limit == 0):
            total_range = len(data)
        else:
            total_range = limit

        if(total_range + offset >= len(self.data)):
            raise ValueError("total_range + offset (" + str(total_range + offset) + ") is greater than the number of data points available in data (" + len(self.data) + ")")
        if(total_range < self.input):
            raise ValueError("total_range (" + str(total_range) + ") cannot be less than the input values for the network (" + self.input + ")")
        self.x, self.y, self.z = range(total_range), np.array(self.data[offset:offset + total_range]), np.ones(total_range)
        # Stack the y values in order to send through to neural network
        n = np.lib.pad(self.y, (self.input, self.input), 'edge')
        temp = []
        for i in range(len(n)-self.input):
            next_x = n[i:i+self.input]
            temp.append((next_x - np.min(next_x))/(np.max(next_x)-np.min(next_x)))
        temp = np.array(temp)
        # Run network
        output = self.sess.run(self.run, feed_dict={self.x_placeholder:temp})
        output = [((n[0] - n[1]) - threshold_min)/(threshold_max - threshold_min) for n in output]
        # Assign averaged output to network
        self.z = np.zeros(len(self.y))
        output = np.array(output)
        for i in range(len(n)-self.input*2):
            self.z[i] =  np.mean(output[i:i+self.input], axis=0)
        return self.x, self.y, self.z

    '''
    Outputs a heatmap using the calculated x, y, and z values
    green = Normal beat
    red = Abnormal beat
    '''
    def graph_output(self):
        plt.style.use('dark_background')
        plot_signal(self.x, self.y, self.z)
        plt.show()


if __name__ == '__main__':
    # parameters
    inputval = 900
    runlimit = 10000
    filename = "1131314/new9.txt"
    modelname = "ann"
    # load data and model
    classifier  = Classifier()
    classifier.load_data(filename)
    classifier.load_model(modelname, inputval)
    classifier.run_model(10000)
    # Create a graph of the final output
    classifier.graph_output()
