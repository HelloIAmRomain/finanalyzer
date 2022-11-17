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
        line_number = tf.range(max_line_number, delta = 30)
        file_number = tf.repeat([i], repeats = len(line_number))
        
        coords.append(tf.stack((file_number, line_number), axis=1))
    
    concat = tf.concat(coords,axis=0)
    
    return concat


# make tf.function accept tensor as input
def create_batch_old(points_list, files_obj, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs):
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
        batch_input.append(np.frombuffer(files_obj[file_number].read(length_input_timeseries * 4 * (n_inputs)), dtype=np.float32))
        # Pointer to the file (output)
        files_obj[file_number].seek(4 * n_inputs * length_files[file_number] + (line_number + length_input_timeseries - length_output_timeseries) * 4 * n_outputs)
        # Read the lines (output)
        batch_output.append(np.frombuffer(files_obj[file_number].read(length_output_timeseries * 4 * n_outputs), dtype=np.float32))
    return np.array(batch_input), np.array(batch_output)


"""
def create_batch(points_list, files_array, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs):
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        inputs = files_array[file_number][line_number:line_number+length_input_timeseries, 0:n_inputs]
        batch_input.append(inputs)
        # Pointer to the file (output)
        # Read the lines (output)
        batch_output.append(files_array[file_number][line_number+length_input_timeseries, n_inputs:])
    return np.array(batch_input), np.array(batch_output)
"""


def create_batch(points_list, files_array, length_files, length_input_timeseries, length_output_timeseries, n_inputs, n_outputs, lines_keep):
    batch_inputs = [[] for model in lines_keep]
    batch_output = []
    for file_number, line_number in points_list:
        # Read the lines (input)
        inputs = files_array[file_number][line_number:line_number+length_input_timeseries, 0:n_inputs]
        for i in range(len(lines_keep)) :
            batch_inputs[i].append(inputs[lines_keep[i]])
        # Pointer to the file (output)
        # Read the lines (output)
        batch_output.append(files_array[file_number][line_number+length_input_timeseries, n_inputs:])
        
    for i in range(len(batch_inputs)) :
        batch_inputs[i] = np.array(batch_inputs[i])
        # print(batch_input.shape)
        
    return batch_inputs, np.array(batch_output)

def fibonacci(n):
    out = 1/np.sqrt(5) * (((1+np.sqrt(5))/2)**n - ((1-np.sqrt(5))/2)**n)
    return np.round(out).astype(int)

def dict_lines_keep():
    """
    Create a list of the lines to keep in the input data
    The max window is the farthest point in the past that we want to use
    """
    list_output = {}
    # # All the lines to keep
    # # list_output.append(np.arange(-max_window, 0))
    # One line out of ten
    # list_output["1_out_10"] = np.arange(-10_000, 0, 10)
    # fiblist=[]
    # for n in range(2, 22):
    #     fiblist.append(-fibonacci(n))
    # fiblist.reverse()
    # list_output["fibonacci"] = np.array(fiblist)
    # List with linear augmentation of distance between points (1, 2, 4, 7, 11, 16, 22...)
    linList = [-1]
    for n in range(1, 140):
        linList.append(linList[-1]-n)
    linList.reverse()
    list_output["linear"] = np.array(linList)
    return list_output

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

list_keep = dict_lines_keep()

n_inputs = 1
n_outputs = 1
input_shape = 10_000
max_shape = -int(min(min(i) for i in list_keep.values()))
print("max_shape", type(max_shape))
output_shape = 1
batch_size = 1_024
epochs = 200

models_topology = [[[1024,"relu"],[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"sigmoid"]],
                   [[1024,"relu"],[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"sigmoid"]],
                   [[1024,"relu"],[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"sigmoid"]],
                   [[1024,"relu"],[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"sigmoid"]],
                   [[1024,"relu"],[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"relu"],[2,"sigmoid"]],
                   [[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"sigmoid"]],
                   [[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"sigmoid"]],
                   [[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"sigmoid"]],
                   [[512,"relu"],[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"relu"],[2,"sigmoid"]],
                   [[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"sigmoid"]],
                   [[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"sigmoid"]],
                   [[256,"relu"],[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"relu"],[2,"sigmoid"]],
                   [[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"sigmoid"]],
                   [[128,"relu"],[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"relu"],[2,"sigmoid"]],
                   [[64,"relu"],[32,"relu"],[16,"relu"],[8,"relu"],[4,"relu"],[2,"sigmoid"]]                   
                   ] 




# models_topology = [[[512,"relu"],[256,"relu"],[128,"sigmoid"]]]

path_BDD_folder = "../BDD/20221027/"
# path_BDD_folder = "../BDD/BDD_test/"

path_working_folder = "./" + time.strftime('%Y_%m_%d_%H_%M_%S') + "/"
bin_files_folder = path_working_folder + "Bin_Files/"
mat_files_folder = path_working_folder + "Matlab/"

# Create the working directories
os.mkdir(path_working_folder)
os.mkdir(bin_files_folder)
os.mkdir(mat_files_folder)

# Copy the python files to directory
shutil.copyfile("FF_linear_topologies.py", path_working_folder+"FF_linear_topologies.py")
# shutil.copyfile("test.py", path_working_folder+"test.py")

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
        files_array.append(file_array)
        
# print(train_bin_files)

print("Time to create the binary files:", time_init + time.time())


###################################################################################################################
# Create the models
###################################################################################################################

models = []
models_path = []
models_name = []


for topology in models_topology :
    for key in list_keep :
        model = build_model(input_shape=(len(list_keep[key]),), output_shape=output_shape, topology=topology)
        models.append(model)
        model_path = path_working_folder + "FF_" + key + "_" + str(len(topology)) + "layers"
        for layer in topology :
            model_path += "_" + str(layer[0])
        os.mkdir(model_path)
        os.mkdir(model_path + "/Models")
        models_path.append(model_path+"/Models")

    
models_loss_fn = [tf.keras.losses.MeanSquaredError() for model in models]
models_optimizer = [tf.keras.optimizers.Adam(learning_rate=1e-4) for model in models]
models_train_acc_metric = [tf.keras.metrics.MeanSquaredError() for model in models]
models_val_acc_metric = [tf.keras.metrics.MeanSquaredError() for model in models]

# Test parallelisation entraînement

models_data = []
for i in range(len(models)) :
    models_data.append([models[i],models_loss_fn[i],models_optimizer[i],models_train_acc_metric[i]])

with open (path_working_folder + "/log_train.csv", "w") as file:
    columns = ";".join(["epoch"] + [str(model_name) for model_name in models_name])
    file.write(columns)
    file.write("\n")
    
with open (path_working_folder + "/log_val.csv", "w") as file:
    columns = ";".join(["epoch"] + [str(model_name) for model_name in models_name])
    file.write(columns)
    file.write("\n")

###################################################################################################################
# Create the coordinates and prepare Training and Validation sets
###################################################################################################################

list_coords = create_list_possible_points(length_files, max_shape, output_shape)
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

###################################################################################################################
# Train the model
###################################################################################################################


t_train_total = -time.time()




min_val_acc = [1 for model in models]

for epoch in range(epochs):
    t_train = 0
    t_epoch = -time.time()
    print(f"Epoch {epoch + 1} / {epochs}")
    models_train_loss = [0 for model in models]
    # Shuffle the data
    train_coords = tf.random.shuffle(train_coords)
    # Create the batches
    train_dataset = tf.data.Dataset.from_tensor_slices(train_coords)
    train_dataset = train_dataset.batch(batch_size)
    n_batch = 0
    with tqdm.tqdm(
            bar_format="{n_fmt}/" + str(len(train_dataset)) + " | Elapsed: {elapsed} | {rate_fmt} | Train Loss: {postfix}",
            postfix=models_train_loss,
            ) as t:
        
        for batch_coords in train_dataset:
            # n_batch += 1
            batch_coords = batch_coords.numpy()
            
            
            batch_inputs, batch_outputs = create_batch(points_list=batch_coords, 
                                                       files_array=files_array, length_files=length_files, 
                                                       n_inputs=n_inputs, n_outputs=n_outputs,length_input_timeseries=max_shape, 
                                                       length_output_timeseries=output_shape,
                                                       lines_keep=list(list_keep.values()))
            for i in range(len(models)) :
                
                t_train -= time.time()
                models_train_loss[i] = train_step(model=models[i], x_batch_train=batch_inputs[0], y_batch_train=batch_outputs, 
                                                  loss_fn=models_loss_fn[i], optimizer=models_optimizer[i], 
                                                  train_acc_metric=models_train_acc_metric[i])
                t_train += time.time()
                
            t.postfix = [models_train_loss[i].numpy() for i in range(len(models))]
            t.update()
        
    
    print("Epoch Train times :", t_epoch + time.time(), "Batch creation :", t_epoch + time.time()-t_train, "Training", t_train)
    # Display metrics at the end of each epoch.
    
    train_acc = [models_train_acc_metric[i].result().numpy() for i in range(len(models))]
    
    with open (path_working_folder + "/log_train.csv", "a") as file:
        columns = ";".join([str(epoch+1)] + [str(acc) for acc in train_acc])
        file.write(columns)
        file.write("\n")
        
    # print("Training MSE over epoch: %.8f" % (float(train_acc),))
    print("Training MSE over epoch:",train_acc)
    # Reset training metrics at the end of each epoch
    for train_acc_metric in models_train_acc_metric :
        train_acc_metric.reset_states()
    
    
    
    # Validation
    t_val_start = -time.time()
    t_val = 0
    for batch_coords in tqdm.tqdm(val_dataset):
        n_batch += 1
        t_batch = -time.time()
        batch_coords = batch_coords.numpy()
        t_val -= time.time()
        
        batch_inputs, batch_outputs = create_batch(points_list=batch_coords, files_array=files_array, 
                                                   length_files=length_files, n_inputs=n_inputs, n_outputs=n_outputs,
                                                   length_input_timeseries=max_shape, length_output_timeseries=output_shape,
                                                   lines_keep=list(list_keep.values()))
        for i in range(len(models)) :
            
            val_step(model=models[i], x_batch_val=batch_inputs[0], y_batch_val=batch_outputs, val_acc_metric=models_val_acc_metric[i])
            
        t_val += time.time()
    
    val_acc = [models_val_acc_metric[i].result().numpy() for i in range(len(models))]
    
    with open (path_working_folder + "/log_val.csv", "a") as file:
        columns = ";".join([str(epoch+1)] + [str(acc) for acc in val_acc])
        file.write(columns)
        file.write("\n")
        
    # print("Training MSE over epoch: %.8f" % (float(train_acc),))
    print("Epoch Val times :", t_val_start + time.time(), "Batch creation :", t_val_start + time.time()-t_val, "Validation", t_val)
    print("Validation MSE over epoch:",val_acc)
    
    # Save models
    for i in range(len(models)) :
        if val_acc[i] < min_val_acc[i] :
            models[i].save(models_path[i] + "/model_" + str(epoch+1) + "_" + str(val_acc[i]) + ".h5")
            min_val_acc[i] = val_acc[i]
    # Reset training metrics at the end of each epoch
    for val_acc_metric in models_val_acc_metric :
        val_acc_metric.reset_states()
        
    print("Epoch time :", t_epoch + time.time())
    print("-----------------------")
    
for i in range(len(models)) :
    models[i].save(models_path[i] + "/model_" + str(epochs) + "_" + str(val_acc[i]) + ".h5")
    

print("Total time :", t_train_total + time.time())


