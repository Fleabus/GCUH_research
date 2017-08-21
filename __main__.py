from setup import setup_data
import numpy as np
#from rnn import RNN
from data_formatter import Data_Formatter
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_formatter = setup_data()
    #data_formatter.load_formatted_data("900_no_noise_binary_equalize")st
    print(len(data_formatter.x))
    print(len(data_formatter.y))
    #data_formatter.noise_generator(0.01)
    data_formatter.equalize_data()

    data_formatter.save_formatted_data("900_no_noise_binary_equalize")
    #data_formatter.shuffle()
    #data_formatter.center_vertical()
    #data_formatter.average_in_window(50, 50)
    '''
    norm = []
    abnorm = []
    for i in range(len(data_formatter.x)):
        if(data_formatter.y[i][0] == 1):
            norm.append(data_formatter.x[i])
        else:
            abnorm.append(data_formatter.x[i])

    plt.plot(norm[0], color="green", alpha=1)
    plt.plot(abnorm[0], color="red", alpha=1)
    plt.show()
    '''
    #data_formatter.save_formatted_data("400_no_noise_binary_equalize")
