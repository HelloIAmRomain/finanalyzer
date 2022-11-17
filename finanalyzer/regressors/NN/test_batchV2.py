import os
import shutil
import sys
import numpy as np
from read_data import read_data
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import gc
import tqdm

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
    for filename in list_files:
        _, inputs, outputs = read_data(path + filename, n_inputs, n_outputs)
        max_value_inputs = np.max(inputs)
        max_value_outputs = np.max(outputs)
    return max_value_inputs, max_value_outputs


def convert_mat_to_bin(path_in, path_out, filename, output_filename, norm_value_inputs, norm_value_outputs , n_inputs=1, n_outputs=1):
    """
    Convert a matlab file to a binary file with a length of 4 bytes per value
    """
    _, inputs, outputs = read_data(path_in + filename, n_inputs, n_outputs)

    inputs = inputs.T / norm_value_inputs
    outputs = outputs.T / norm_value_outputs
    
    # Concatenate the data as a 2D array
    data = np.concatenate((inputs, outputs), axis=0)
    # Convert to float32 with a length of 4 bytes
    data = data.astype(np.float32)
    # Save to binary file
    data.tofile(path_out + output_filename)
    
    return len(inputs)


def convert_mat_to_np(path_in, path_out, filename, output_filename, norm_value_inputs, norm_value_outputs , n_inputs=1, n_outputs=1):
    """
    Convert a matlab file to a binary file with a length of 4 bytes per value
    """
    _, inputs, outputs = read_data(path_in + filename, n_inputs, n_outputs)

    inputs = inputs.T / norm_value_inputs
    outputs = outputs.T / norm_value_outputs
    
    # Concatenate the data as a 2D array
    data = np.concatenate((inputs, outputs), axis=1)
    # Convert to float32 with a length of 4 bytes
    data = data.astype(np.float32)    
    return data


def read_bin_data(path, filename, n_inputs=1, n_outputs=1, length_file=5_000_000):
    """
    Read a binary file and return the input and output data
    """
    data = np.fromfile(path + filename, dtype=np.float32)
    inputs = data[:length_file * n_inputs]
    outputs = data[length_file * n_inputs:]
    return inputs, outputs


@tf.function
def create_list_possible_points(length_files, length_input_timeseries, length_output_timeseries):
    """
    Create a list of all the possible points in the data
    """
    coords = []
    for i in range(len(length_files)) :
        max_line_number = length_files[i] - max(length_input_timeseries, length_output_timeseries)
        file_number = tf.repeat([i], repeats = max_line_number)
        line_number = tf.range(max_line_number)
        coords.append(tf.stack((file_number, line_number), axis=1))
    
    concat = tf.concat(coords,axis=0)
    
    return concat


# def create_batch(points_list, files_obj, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs):
#     """
#     Create a batch of data

#     points_list is a numpy array that contains the file number and the line number
#     example [[3, 542587], [1, 547365], ...]
#     File number 1 corresponds to the file Batch_0001.mat
#     """
#     batch_input = []
#     batch_output = []
#     for file_number, line_number in points_list:
#         # file_number = int(file_number)
#         # line_number = int(line_number)
#         # Pointer to the file (input)
#         files_obj[file_number].seek(line_number * 4 * n_inputs)
#         # Read the lines (input)
#         batch_input.append(np.frombuffer(files_obj[file_number].read(length_input_timeseries * 4 * (n_inputs)), dtype=np.float32))
#         # Pointer to the file (output)
#         files_obj[file_number].seek(4 * n_inputs * length_files[file_number] + (line_number + length_input_timeseries - length_output_timeseries) * 4 * n_outputs)
#         # Read the lines (output)
#         batch_output.append(np.frombuffer(files_obj[file_number].read(length_output_timeseries * 4 * n_outputs), dtype=np.float32))
#     return np.array(batch_input), np.array(batch_output)


def create_batch_2(points_list, files_obj, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs):
    """
    Create a batch of data

    points_list is a numpy array that contains the file number and the line number
    example [[3, 542587], [1, 547365], ...]
    File number 1 corresponds to the file Batch_0001.mat
    """
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        # file_number = int(file_number)
        # line_number = int(line_number)
        # Pointer to the file (input)
        files_obj[file_number].seek(line_number * 4 * n_inputs)
        # Read the lines (input)
        values = np.frombuffer(files_obj[file_number].read(length_input_timeseries * 4 * (n_inputs)), dtype=np.float32)
        # Get one value every milisecond then all values of the last milisecond
        inputs = np.concatenate((values[0:-1000*n_inputs:100], values[-1000*n_inputs:]), axis=0)
        batch_input.append(inputs)
        # Pointer to the file (output)
        files_obj[file_number].seek(4 * n_inputs * length_files[file_number] + (line_number + length_input_timeseries - length_output_timeseries) * 4 * n_outputs)
        # Read the lines (output)
        batch_output.append(np.frombuffer(files_obj[file_number].read(length_output_timeseries * 4 * n_outputs), dtype=np.float32))
    return np.array(batch_input), np.array(batch_output)


def create_batch_3(points_list, files_obj, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs, lines_keep):
    """
    Create a batch of data

    points_list is a numpy array that contains the file number and the line number
    example [[3, 542587], [1, 547365], ...]
    File number 1 corresponds to the file Batch_0001.mat
    """
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        files_obj[file_number].seek(line_number * 4 * n_inputs)
        # Read the lines (input)
        values = np.frombuffer(files_obj[file_number].read(length_input_timeseries * 4 * (n_inputs)), dtype=np.float32)
        inputs = values[lines_keep]
        batch_input.append(inputs)
        # Pointer to the file (output)
        files_obj[file_number].seek(4 * n_inputs * length_files[file_number] + (line_number + length_input_timeseries - length_output_timeseries) * 4 * n_outputs)
        # Read the lines (output)
        batch_output.append(np.frombuffer(files_obj[file_number].read(length_output_timeseries * 4 * n_outputs), dtype=np.float32))
    return np.array(batch_input), np.array(batch_output)


def create_batch_4(points_list, files_array, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs, lines_keep):
    """
    Create a batch of data

    points_list is a numpy array that contains the file number and the line number
    example [[3, 542587], [1, 547365], ...]
    File number 1 corresponds to the file Batch_0001.mat
    """
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        # Read the lines (input)
        batch_input.append(files_array[file_number][line_number:line_number+length_input_timeseries, 0:n_inputs][lines_keep])
        # Pointer to the file (output)
        # Read the lines (output)
        batch_output.append(files_array[file_number][line_number+length_input_timeseries, n_inputs:])
    
    return np.array(batch_input), np.array(batch_output)




def list_val(max):
    out=[1]
    i=1
    while out[-1]+i<max:
        out.append(out[-1]+i)
        i+=1
    out = np.array(out) * -1
    out.sort
    return out



# @tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer, train_acc_metric):
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


def val_step(model, x_batch_val, y_batch_val, val_acc_metric) :
    val_logits = model(x_batch_val, training=False)
    val_acc_metric.update_state(y_batch_val, val_logits)


def build_model(input_shape, output_shape, topology):
    """
    Build a FF to predict a time series from another one

    Input: numpy array, dimension(n_features, n_samples)
    Output: numpy array, dimension(n_features, n_samples)
    """
    model = Sequential()
    
    # Couche d'entrée
    model.add(Dense(topology[0][0], input_shape=input_shape))
    model.add(Activation(topology[0][1]))
    
    # Autres couches
    for layer in topology[1:] :
        model.add(Dense(layer[0], input_shape=input_shape))
        model.add(Activation(layer[1]))
        
    # Couche de sortie
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    
    return model



###################################################################################################################
# Variables
###################################################################################################################

n_inputs = 1
n_outputs = 1
input_shape = 10_000
output_shape = 1
batch_size = 2_000
epochs = 50

models_topology = [[[512,"relu"],[256,"sigmoid"]]]


# models_topology = [[[512,"relu"],[256,"relu"],[128,"sigmoid"]]]

path_BDD_folder = "../BDD/BDD_test/"
# path_BDD_folder = "../BDD/BDD_test/"

path_working_folder = "./" + time.strftime('%Y_%m_%d_%H_%M_%S') + "/"
bin_files_folder = path_working_folder + "Bin_Files/"
mat_files_folder = path_working_folder + "Matlab/"

# Create the working directories
os.mkdir(path_working_folder)
os.mkdir(bin_files_folder)
os.mkdir(mat_files_folder)

# Copy the python files to directory
shutil.copyfile("FF_multiple_models.py", path_working_folder+"FF_multiple_models.py")
shutil.copyfile("test.py", path_working_folder+"test.py")

# Copy the files that are used to create mat files
# shutil.copyfile("../DATA/matlab_multithreaded_workers_v3.py", path_working_folder+"matlab_multithreaded_workers_v3.py")
# shutil.copyfile("../DATA/SIMPLE/MATLAB/outputs/inductance_saturante/_Batch_parameters.txt", path_working_folder+"_Batch_parameters.txt")
# shutil.copyfile("../DATA/SIMPLE/MATLAB/outputs/inductance_saturante/_Test_parameters.txt", path_working_folder+"_Test_parameters.txt")

time_init = -time.time()

###################################################################################################################
# Create the binary files
###################################################################################################################

mat_files = [f for f in os.listdir(path_BDD_folder) if f.endswith('.mat')]
max_value_inputs, max_value_outputs = get_max_value(path_BDD_folder, mat_files, n_inputs=1, n_outputs=1)
print("max value inputs:", max_value_inputs)
print("max value outputs:", max_value_outputs)


length_files = []
files_array = []

for file in mat_files:
    output_filename = file.replace(".mat", ".bin")
    file_array = convert_mat_to_np(path_in=path_BDD_folder, path_out=bin_files_folder, filename=file, output_filename=output_filename, 
                                     norm_value_inputs=max_value_inputs, norm_value_outputs=max_value_outputs, n_inputs=1, n_outputs=1)
    
    # print(output_filename, size_file)
    
    if file.startswith('train') :
        length_files.append(len(file_array))
        print(file, file_array.shape)
        files_array.append(file_array)
    
    



###################################################################################################################
# Create the coordinates and prepare Training and Validation sets
###################################################################################################################

list_coords = create_list_possible_points(length_files, input_shape, output_shape)
print("Number of coordinates =",len(list_coords))

list_coords = tf.random.shuffle(list_coords)
nb_coords_val = int(len(list_coords)*0.1)

val_coords = list_coords[:nb_coords_val]
train_coords = list_coords[nb_coords_val:]
print("Number of Validation coordinates =",len(val_coords))
print("Number of Training coordinates =",len(train_coords))

val_dataset = tf.data.Dataset.from_tensor_slices(val_coords)
val_dataset = val_dataset.batch(batch_size)

del list_coords
del val_coords
gc.collect()



list_keep_lines = list_val(10_000)



t_train_total = -time.time()

for epoch in range(epochs):
    t_train = 0
    t_epoch = -time.time()
    print(f"Epoch {epoch + 1} / {epochs}")
    # Shuffle the data
    train_coords = tf.random.shuffle(train_coords)
    # Create the batches
    train_dataset = tf.data.Dataset.from_tensor_slices(train_coords)
    train_dataset = train_dataset.batch(batch_size)
    n_batch = 0
    for batch_coords in tqdm.tqdm(train_dataset):
        # n_batch += 1

        batch_coords = batch_coords.numpy()
        batch_inputs, batch_outputs = create_batch_4(points_list=batch_coords, files_array=files_array, 
                                                     length_files=length_files, n_inputs=n_inputs, n_outputs=n_outputs,
                                                     length_input_timeseries=input_shape, length_output_timeseries=output_shape,
                                                     lines_keep=list_keep_lines,
                                                     )
    t_epoch_spent = t_epoch + time.time()
    print(f"epoque: {t_epoch_spent}s (batch: {t_epoch_spent/len(train_dataset)})")
        
