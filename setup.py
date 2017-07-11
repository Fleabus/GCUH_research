import numpy as np
from data_formatter import Data_Formatter
from read_data import loadAndSlice

def choice_result(choice):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
    choice = choice.lower()
    if choice in yes:
       return True
    elif choice in no:
       return False
    else:
       return False

def load_files(type):
    print("Loading files... ")
    x, y = loadAndSlice(type, "mitdb")
    return x, y

def setup_data(type):
    data_x, data_y = [], []
    try:
        data_x = np.load("data/features_" + type + ".npy")
        data_y = np.load("data/labels_" + type + ".npy")
    except IOError:
        print("Training and testing data of type", type, "does not already exist.")
        choice = input("Would you like to setup a training and testing set? [Y/n]\n")
        choice = choice_result(choice)
        if(choice):
            data_x, data_y = load_files(type)
            print("Saving data...")
            np.save("data/features_" + type, data_x)
            np.save("data/labels_" + type, data_y)
    finally:
        print("Returning data...")
        data_formatter = Data_Formatter()
        data_formatter.assignData(data_x, data_y)
        return data_formatter
