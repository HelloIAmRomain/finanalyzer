# RNN regressor using scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sqlite3
import argparse
import os


def read_data(id_company, database):
    """Reads the data from the database and returns a pandas dataframe"""
    if type(id_company) != int:
        raise ValueError("id must be an integer")
    conn = sqlite3.connect(database)
    result = conn.execute(f'SELECT close FROM valuesFinHistory where namesId = {id_company}')
    # Convert from list of tuples to pandas dataframe
    df = pd.DataFrame(result.fetchall())
    conn.close()
    return df


def all_ids(database):
    """Returns a list with all the ids in the database"""
    conn = sqlite3.connect(database)
    result = conn.execute('SELECT DISTINCT namesId FROM valuesFinHistory')
    # Convert from list of tuples to pandas dataframe
    ids = list(result.fetchall())
    ids = [i[0] for i in ids]
    conn.close()
    return ids


def train_test(df, proportion_test):
    """Splits the data into training and testing data
    
    Arguments:
        df {pandas dataframe} -- Dataframe with the data
        proportion_test {float} -- Proportion of the data to be used for testing (0.0-1.0) default 0.15
        
    Returns:
        test {pandas dataframe} -- Dataframe with the testing data
        train {pandas dataframe} -- Dataframe with the training data
    """
    train, test = train_test_split(df, test_size=proportion_test)
    return train, test


def prep_data(df, size_in, size_out):
    """Generates two arrays to split the data into input and output
    
    Arguments:
        df {pandas dataframe} -- Dataframe with the data
        size_in {int} -- Size of the neural network input
        size_out {int} -- Size of the neural network output
        
    Returns:
        X [np.array] -- Array with the input data
        y [np.array] -- Array with the output data
    """
    if size_in + size_out > len(df):
        raise ValueError("Size of input and output is too big for the data")
    X = []
    y = []
    for i in range(len(df) - size_in - size_out + 1):
        X.append(np.concatenate(df.iloc[i:i + size_in].values))
        y.append(np.concatenate(df.iloc[i + size_in:i + size_in + size_out].values))
    return np.array(X), np.array(y)


def create_neural_network(size_in, size_out, hidden_layers=(100, 100, 100)):
    """Creates a neural network using keras
    
    Arguments:
        size_in {int} -- Size of the neural network input
        size_out {int} -- Size of the neural network output
        hidden_layers {tuple} -- Tuple with the number of neurons in each hidden layer
        
    Returns:
        model {keras model} -- Keras model of the neural network
    """
    model = Sequential()
    model.add(Input(shape=(size_in,)))
    for i in range(len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='relu', use_bias=True))
    model.add(Dense(size_out, activation='linear', use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def train_neural_network(model, train_X, train_y, verbose):
    """Trains the keras model

    Arguments:
        model {keras model} -- Keras model of the neural network
        train_X {np.array} -- Array with the input data
        train_y {np.array} -- Array with the output data

    Returns:
        model {keras model} -- Trained keras model
    """
    model.fit(train_X, train_y, epochs=100, batch_size=10, verbose=verbose)
    # Save the model
    return model


def test_neural_network(model, test_X, test_y, plot):
    """Tests the neural network

    Arguments:
        model {keras model} -- Trained keras model
        test_X {np.array} -- Array with the input data
        test_y {np.array} -- Array with the output data
        plot {bool} -- If true, plots the results

    Returns:
        predictions {np.array} -- Array with the predictions
        test_y {np.array} -- Array with the actual values
    """
    predictions = model.predict(test_X)
    accuracy = model.evaluate(test_X, test_y)
    meansquarederror = mean_squared_error(test_y, predictions)
    meanabsoluteerror = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    if plot:
        # show a plot of the actual values (blue) vs the predicted values (red)
        plt.plot(test_y, label='Actual', color='blue', linewidth=1, linestyle='solid')
        plt.plot(predictions, label='Predicted', color='red')
        plt.show()
    return accuracy, meansquarederror, meanabsoluteerror, r2


def welcome(id_company, size_in, size_out, hidden_layers, proportion_test):
    """Prints a welcome message"""
    print(f"Welcome to the neural network regressor")
    print(f"Using data from: {id_company}")
    print(f"Size of input: {size_in}")
    print(f"Size of output: {size_out}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Proportion of data used for testing: {proportion_test}")


def main():
    parser = argparse.ArgumentParser(description='Neural network regressor')
    parser.add_argument('-id', help="List of ids to use separated bu spaces", nargs='+', type=int)
    parser.add_argument('-size_in', '-i', type=int, help='Size of the input')
    parser.add_argument('-size_out', '-o', type=int, help='Size of the output')
    parser.add_argument('-hidden_layers', '-hl', type=int, nargs='+',
                        help='Number of neurons in each hidden layer (separated by spaces), default (100, 100, 100)',
                        default=(100, 100, 100))
    parser.add_argument('-proportion_test', '-pt', type=float,
                        help='Proportion of the data to be used for testing (default 0.15)', default=0.15)
    parser.add_argument('-save_model', type=bool, help='Save the model (default True)', default=True)
    parser.add_argument('-load_model', type=bool, help='Load the model (default False -> overwrites)', default=False)
    parser.add_argument('-model_name', type=str, help='Name of the model (default dense_model.h5)',
                        default='dense_model.h5')
    parser.add_argument('-plot', type=bool, help='Plot the actual values vs the predicted values (default True)',
                        default=True)
    parser.add_argument('-verbose', type=bool, help='Verbose (default True)', default=True)
    parser.add_argument('-database_name', '-db', type=str, help='Name of the database (default database.db)',
                        default="financial_data.db")
    args = parser.parse_args()
    print(args)

    # Define constants
    id_companies = args.id if args.id else 0
    size_in = args.size_in if args.size_in else 30
    size_out = args.size_out if args.size_out else 10
    hidden_layers = args.hidden_layers
    proportion_test = args.proportion_test
    save_the_model = args.save_model
    load_the_model = args.load_model
    model_name = args.model_name
    plot = args.plot
    verbose = args.verbose
    database_name = args.database_name

    # Print a welcome message
    if verbose:
        welcome(id_companies, size_in, size_out, hidden_layers, proportion_test)

    if not id_companies:
        print("no company id provided, working with all the companies")
        # Get the data from the database
        id_companies = all_ids(database_name)
    for id_company in id_companies:
        # Read the data from the database
        df = read_data(id_company, database_name)
        # Split the data into training and testing data
        train, test = train_test(df, proportion_test)
        # Prepare the data for the neural network
        train_X, train_y = prep_data(train, size_in, size_out)
        test_X, test_y = prep_data(test, size_in, size_out)
        # Create the neural network if it doesn't exist or if the user wants to create a new one
        if os.path.isfile("dense_model.h5") and load_the_model:
            model = load_model("dense_model.h5")
        else:
            model = create_neural_network(size_in, size_out, hidden_layers)
        # Train the neural network
        model = train_neural_network(model, train_X, train_y, verbose)
        # Test the neural network
        accuracy, meansquarederror, meanabsoluteerror, r2 = test_neural_network(model, test_X, test_y, plot)
        if verbose:
            print("Accuracy: ", accuracy)
            print("Mean squared error: ", meansquarederror)
            print("Mean absolute error: ", meanabsoluteerror)
            print("R2: ", r2)
    # Save the model
    if save_the_model:
        save_model(model, model_name)
    print("Done")


if __name__ == "__main__":
    main()
