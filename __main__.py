from setup import setup_data
from rnn import RNN

if __name__ == "__main__":
    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())

    data_formatter.shuffle()
    data_formatter.equalize_data()
    data_formatter.split_training_testing(0.3)

    rnn = RNN(data_formatter, learning_rate=0.5, epochs=1000,
                learning_rate_reduction=0.5, reduction_index=1,
                batch_size=240)
    rnn.setup()
    rnn.train()
