import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def read_data(filename, input_size, output_size):
    """
    Reads data from a .mat file and returns it as a numpy array

    file has a variable called 'database' which is a 2D array with columns:
        [time, input1, input2, ..., inputN, output1, output2, ..., outputM]


    convert [t1, input1, output1]
            [t2, input2, output2]
            ...............
            [tN, inputN, outputN]
    to
            [t1, t2, ..., tN]
            [input1, input2, ..., inputN]
            [output1, output2, ..., outputN]
    """
    data = sio.loadmat(filename)
    database = data['database']
    time = np.array(database[:, 0])
    # Reshape the input and output data to be 2D arrays
    inputs = np.array(database[:, 1:input_size+1])
    outputs = np.array(database[:, input_size+1:input_size+output_size+1])
    return time.T, inputs.T, outputs.T
    