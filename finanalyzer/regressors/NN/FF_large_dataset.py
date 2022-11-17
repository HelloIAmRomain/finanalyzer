import os
import numpy as np
import os
from read_data import read_data
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import gc
import csv
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try: 
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    

def get_max_value(path, list_files, n_inputs=1, n_outputs=1):
    """
    Get the maximum value of the data
    """
    max_value = 0
    for filename in list_files:
        _, inputs, outputs = read_data(path + filename, n_inputs, n_outputs)
        max_value = max(max_value, np.max(inputs), np.max(outputs))
    return max_value


def convert_mat_to_bin(path_in, path_out, filename, output_filename, norm_value, n_inputs=1, n_outputs=1):
    """
    Convert a matlab file to a binary file with a length of 4 bytes per value
    """
    _, inputs, outputs = read_data(path_in + filename, n_inputs, n_outputs)

    inputs = inputs.T / norm_value
    outputs = outputs.T / norm_value
    
    # Concatenate the data as a 2D array
    data = np.concatenate((inputs, outputs), axis=0)
    # Convert to float32 with a length of 4 bytes
    data = data.astype(np.float32)
    # Save to binary file
    data.tofile(path_out + output_filename)


def read_bin_data(path, filename, n_inputs=1, n_outputs=1, length_file=5_000_000):
    """
    Read a binary file and return the input and output data
    """
    data = np.fromfile(path + filename, dtype=np.float32)
    inputs = data[:length_file * n_inputs]
    outputs = data[length_file * n_inputs:]
    return inputs, outputs


@tf.function
def create_list_possible_points(length_file, number_files, length_input_timeseries, length_output_timeseries):
    """
    Create a list of all the possible points in the data
    """
    max_line_number = length_file - max(length_input_timeseries, length_output_timeseries) + 1
    file_number = tf.repeat(tf.range(1, number_files + 1), tf.constant([max_line_number]))
    line_number = tf.tile(tf.range(1, max_line_number + 1), tf.constant([number_files]))
    concat = tf.stack((file_number, line_number), axis=1)
    return concat


# make tf.function accept tensor as input
def create_batch(points_list, files_obj, length_input_timeseries, length_file=5_000_000, n_inputs=1, n_outputs=1, length_output_timeseries=1):
    """
    Create a batch of data

    points_list is a numpy array that contains the file number and the line number
    example [[3, 542587], [1, 547365], ...]
    File number 1 corresponds to the file Batch_0001.mat
    """
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        file_number = int(file_number)
        line_number = int(line_number)
        # Pointer to the file (input)
        files_obj[file_number].seek(line_number * 4 * n_inputs)
        # Read the lines (input)
        batch_input.append(np.frombuffer(files_obj[file_number].read(length_input_timeseries * 4 * (n_inputs)), dtype=np.float32))
        # Pointer to the file (output)
        files_obj[file_number].seek(4 * n_inputs * length_file + (line_number + length_input_timeseries - length_output_timeseries) * 4 * n_outputs)
        # Read the lines (output)
        batch_output.append(np.frombuffer(files_obj[file_number].read(length_output_timeseries * 4 * n_outputs), dtype=np.float32))
    return np.array(batch_input), np.array(batch_output)


@tf.function
def train_step(model, x_batch_train, y_batch_train):
    """
    passage du batch dans le modèle pour réaliser l'entrainement
    """
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, logits)
    # Gradient
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Mise à jour des poids
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Train metrics
    train_acc_metric.update_state(y_batch_train, logits)
    return loss_value



def build_model(input_shape, output_shape):
    """
    Build a CNN to predict a time series from another one

    Input: numpy array, dimension(n_features, n_samples)
    Output: numpy array, dimension(n_features, n_samples)
    """
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    return model



###################################################################################################################
# Variables
###################################################################################################################

path_mat_data = "../DATA/SIMPLE/MATLAB/outputs/inductance_saturante/"
path_working_folder = "./" + time.strftime('%Y_%m_%d_%H_%M_%S') + "/"
# Create the working directory to save the models
os.mkdir(path_working_folder)

# Copy the python file to directory



mat_files = [f for f in os.listdir(path_mat_data) if f.endswith('.mat')]
size_file = 5_000_000
input_shape = (10_000,)
output_shape = 1
batch_size = 1_000
epochs = 100


###################################################################################################################
# Create the binary files
###################################################################################################################

time_init = -time.time()
max_value = get_max_value(path_mat_data, mat_files, n_inputs=1, n_outputs=1)
print("max value:", max_value)
for file in mat_files:
    output_filename = file.replace(".mat", ".bin")
    convert_mat_to_bin(path_in=path_mat_data, path_out=path_working_folder, filename=file, output_filename=output_filename, norm_value=max_value, n_inputs=1, n_outputs=1)

bin_files = [f for f in os.listdir(path_working_folder) if f.endswith('.bin')]
# Create a list of files objects
files_obj = {}
for file in range(len(bin_files)):
    files_obj[file + 1] = open(path_working_folder + bin_files[file], "rb")
print("Time to create the binary files:", time_init + time.time())


###################################################################################################################
# Create the model
###################################################################################################################

model = build_model(input_shape=input_shape, output_shape=output_shape)
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_acc_metric = tf.keras.metrics.MeanSquaredError()
val_acc_metric = tf.keras.metrics.MeanSquaredError()



###################################################################################################################
# Train the model
###################################################################################################################

list_points = create_list_possible_points(length_file=5_000_000, number_files=len(bin_files), length_input_timeseries=input_shape[0], length_output_timeseries=1)

t_train = -time.time()


for epoch in range(epochs):
    t_epoch = -time.time()
    print(f"Epoch {epoch + 1} / {epochs}")
    # Shuffle the data
    list_points = tf.random.shuffle(list_points)
    # Create the batches
    train_dataset = tf.data.Dataset.from_tensor_slices(list_points)
    train_dataset = train_dataset.batch(batch_size)
    n_batch = 0
    for coord_batch in train_dataset:
        n_batch += 1
        t_batch = -time.time()
        coord_batch = coord_batch.numpy()
        batch_inputs, batch_outputs = create_batch(points_list=coord_batch, files_obj=files_obj, length_file=5_000_000, n_inputs=1, n_outputs=1,length_input_timeseries=input_shape[0], length_output_timeseries=1)
        loss = train_step(model=model, x_batch_train=batch_inputs, y_batch_train=batch_outputs)
        if n_batch % 1000 == 0:
            print(f"\tLoss average (batch {n_batch}): {loss}")
    print("Epoch time :", t_epoch + time.time())
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training MSE over epoch: %.8f" % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    print("-----------------------")
print("Total time :", t_train + time.time())


###################################################################################################################
# Save the model
###################################################################################################################

path_saved_model = path_working_folder + "saved_model.h5"
model.save(path_saved_model)


###################################################################################################################
# Test the model
###################################################################################################################
stop = input("pause")
t_test = -time.time()
x_test = []
y_pred = []
y_test = []

list_points = create_list_possible_points(length_file=5_000_000, number_files=1, length_input_timeseries=input_shape[0], length_output_timeseries=1)
test_dataset = tf.data.Dataset.from_tensor_slices(list_points)
test_dataset = test_dataset.batch(batch_size)
n_batch = 0
for coord_batch in test_dataset :
    n_batch += 1
    t_batch = -time.time()
    coord_batch = coord_batch.numpy()
    batch_inputs, batch_outputs = create_batch(points_list=coord_batch, files_obj=files_obj, length_file=5_000_000, n_inputs=1, n_outputs=1,length_input_timeseries=input_shape[0], length_output_timeseries=1)
    y_pred.append(model.predict_on_batch(batch_inputs))
    y_test.append(batch_outputs)
    x_test.append(batch_inputs[:,-1])
    if n_batch % 1000 == 0:
        print(n_batch,"batch predicted")



print("Total Time Test :",t_test+time.time())
y_test = np.concatenate(y_test,axis = 0)
y_pred = np.concatenate(y_pred,axis = 0)
x_test = np.concatenate(x_test,axis = 0)

# for i in range(1000):
#     x_ = x_test[i:i+input_shape[0]].reshape(1, input_shape[0])
#     y_pred.append(model.predict(x=x_))


# Save prediction to csv (without the )
x_test_save = x_test[-len(y_pred):].T
y_test_save = y_test[-len(y_pred):].T
y_pred_save = y_pred.T
concat_output = np.concatenate((x_test_save, y_test_save, y_pred_save), axis=0)
filename_output = path_working_folder + "output.csv"


np.savetxt(filename_output, concat_output, delimiter=',', newline='\n')
# with open(filename_output, "w", newline="") as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',')
    


print(f"Test MSE: {np.mean(np.abs(y_pred-y_test))}")

# plot result
plt.plot(y_test[:1000], label='Expected inductance')
plt.plot(y_pred[:1000], label='Predicted inductance')
plt.legend()
plt.show()



