from setup import setup_data
import numpy as np
from rnn import RNN
from data_formatter import Data_Formatter
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_formatter = setup_data()
    #data_formatter.noise_generator(0.01)
    print(len(data_formatter.x))
    data_formatter.equalize_data()
    print(len(data_formatter.x))
    data_formatter.save_formatted_data("360_no_noise_binary_equalize")
