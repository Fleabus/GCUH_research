from setup import setup_data
from rnn import RNN

if __name__ == "__main__":
    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())

    data_formatter.shuffle()
    data_formatter.equalize_data()
    data_formatter.split_training_testing(0.3)

    '''
    rnn = RNN(data_formatter)
    rnn.setup()
    rnn.train()
    '''
