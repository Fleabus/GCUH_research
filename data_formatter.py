import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Data_Formatter:
    x, y = [], []
    x_train, y_train = [], []
    x_test, y_test = [], []

    def assign_data(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def shuffle(self):
        s = np.arange(self.x.shape[0])
        np.random.shuffle(s)
        self.x = self.x[s]
        self.y = self.y[s]

    # Ensures all classifications are of equal length
    def equalize_data(self):
        x_values = []
        y_values = []
        classifications = len(self.y[0])
        # Create 2D array consisting of all classifications
        # e.g. 2 classifications = [[list1], [list2]]
        for _ in range(classifications):
            x_values.append([])
            y_values.append([])
        # Cycle through array and assign each to appropriate list
        for i in range(len(self.x)):
            index_point = np.argmax(self.y[i])
            x_values[index_point].append(self.x[i])
            y_values[index_point].append(self.y[i])
        self.x = []
        self.y = []
        # Find the minimum amount of data points in a category
        minAmount = -1
        for i in range(classifications):
            if(len(x_values[i]) < minAmount or minAmount == -1):
                minAmount = len(x_values[i])
        # Cycle through all classifications and reduce to minAmount
        self.x = x_values[0:classifications][0:minAmount]
        self.y = y_values[0:classifications][0:minAmount]
        # Assign back to numpy arrays for processing
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        # Reshape into standard format [total, steps]
        '''
        This currently does not support more than 2 classifications
        '''
        self.x = np.concatenate((self.x[0], self.x[1]), axis=0)
        self.y = np.concatenate((self.y[0], self.y[1]), axis=0)

    def average_in_window(self, window_size, stride=0):
        new_x = []
        for x_val in self.x:
            new_x_val = []
            for i in range(len(x_val)-window_size):
                windowed_list = x_val[i:i+window_size]
                new_x_val.append(sum(windowed_list) / float(len(windowed_list)))
            new_x.append(new_x_val)
        self.x = new_x

    # Splits the dataset based on the testing percentage input
    def split_training_testing(self, testing_percent=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=testing_percent, random_state=42)

    # returns x and y batch
    def get_batch(self, batch_size, iterator = 0, type="train"):
        if(type == "train"):
            if(batch_size*iterator > len(self.x_train)):
                print("Batch index out of range", batch_size*iterator, len(x))
                return np.array([]), np.array([])
            return self.x_train[batch_size*iterator:batch_size*iterator + batch_size], self.y_train[batch_size*iterator:batch_size*iterator + batch_size]
        elif(type == "test"):
            if(batch_size*iterator > len(self.x_test)):
                print("Batch index out of range", batch_size*iterator, len(x))
                return np.array([]), np.array([])
            return self.x_test[batch_size*iterator:batch_size*iterator + batch_size], self.y_test[batch_size*iterator:batch_size*iterator + batch_size]

    def split_into_chunks(self, data, size):
        n = max(data, size)
        return (data[i:i+size] for i in xrange(0, len(data), size))

    def plot_accuracy(self,features, labels):
        plt.style.use('dark_background')
        for i in range(len(features)):
            if(labels[i]== True):
                plt.plot(features[i], color="green",alpha=0.1)
            else:
                plt.plot(features[i], color="red", alpha=0.1)

    def counter(self):
        normal_counter = 0
        abnormal_counter = 0
        for i in range(len(self.y)):
            if(self.y[i][0] == 1):
                normal_counter = normal_counter + 1
            if(self.y[i][1] == 1):
                abnormal_counter = abnormal_counter + 1
        print(normal_counter)
        print(abnormal_counter)
