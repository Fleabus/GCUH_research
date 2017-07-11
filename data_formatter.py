import numpy as np

class Data_Formatter:
    x, y = [], []

    def assignData(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

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
        print(len(x_values))
        print(len(x_values[0]), len(x_values[1]))
