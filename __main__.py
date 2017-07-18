from setup import setup_data
import numpy as np
from rnn import RNN
from data_formatter import Data_Formatter
import tensorflow as tf
import sys

if __name__ == "__main__":
    data_type = input("What lead? (e.g. MLII, V1, V2...):\n")
    data_formatter = setup_data(data_type.upper())

    data_formatter.shuffle()
    data_formatter.equalize_data()
    '''
    x = []
    y = []
    for i in range(20000):
        if(i % 2 == 0):
            x.append(np.zeros((360)))
            y.append([1, 0])
        else:
            x.append(np.ones((360)))
            y.append([0, 1])

    x = np.array(x)
    y = np.array(y)

    data_formatter = Data_Formatter()
    data_formatter.assign_data(x, y)
    '''
    data_formatter.split_training_testing(0.3)


    #Hyperparameters
    epochs = 10
    batch_size = 124

    # Create rnn
    rnn = RNN(learning_rate=0.001)

    with tf.Session() as sess:
        rnn.setup(sess)

        for i in range(epochs):
            epoch_err = 0.0
            # Run training loop
            for j in range(int(len(data_formatter.x_train)/batch_size)):
                batch_x, batch_y = data_formatter.get_batch(batch_size, j, "train")
                # Convert shape [batch * steps] -> [batch * steps * inputs]
                batch_x = np.expand_dims(batch_x, axis=2)
                err, _ = rnn.train(batch_x, batch_y)
                epoch_err = epoch_err + err
                sys.stdout.write("\rEpoch " + str(i + 1) + " training ... {0:.2f}%".format((float((j * batch_size)/len(data_formatter.x_train)))*100))
                sys.stdout.flush()
                prev_epoch_err = epoch_err
            sys.stdout.write("\rEpoch " + str(i + 1) + " training ... complete!")

            total_accuracy = 0.0
            for j in range(int(len(data_formatter.x_test)/batch_size)):
                batch_x, batch_y = data_formatter.get_batch(batch_size, j, "test")
                # Convert shape [batch * steps] -> [batch * steps * inputs]
                batch_x = np.expand_dims(batch_x, axis=2)
                # Run test over batc
                acc, corr = rnn.test(batch_x, batch_y)
                total_accuracy = total_accuracy + acc
            print("\nError:", epoch_err, "\nAccuracy:", total_accuracy / float(len(data_formatter.x_test)/batch_size), "\n")
