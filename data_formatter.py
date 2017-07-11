import numpy as np

class Data_Formatter:
    x, y = [], []

    def assignData(self, x_data, y_data):
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
