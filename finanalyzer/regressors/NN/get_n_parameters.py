# Use trained model to predict the test data in batches

import numpy as np
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import glob
import sys
import math
import gc


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
    batch_input = []
    batch_output = []
    for file_number, line_number in points_list:
        # Read the lines (input)
        batch_input.append(files_array[file_number][line_number:line_number+length_input_timeseries, 0:n_inputs][lines_keep])
        # Pointer to the file (output)
        # Read the lines (output)
        batch_output.append(files_array[file_number][line_number+length_input_timeseries, n_inputs:])
    return np.array(batch_input), np.array(batch_output)


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
    list_output["1_out_10"] = np.arange(-10_001, 0, 10)
    fiblist=[]
    for n in range(2, 22):
        fiblist.append(-fibonacci(n))
    fiblist.reverse()
    list_output["fibonacci"] = np.array(fiblist)
    # List with linear augmentation of distance between points (1, 2, 4, 7, 11, 16, 22...)
    linList = [-1]
    for n in range(1, 140):
        linList.append(linList[-1]-n)
    linList.reverse()
    list_output["linear"] = np.array(linList)
    return list_output


###################################################################################################################
# Parameters
###################################################################################################################
list_keep = dict_lines_keep()

batch_size = 1_024
length_input_timeseries = 10_000
max_input_timeseries = -int(min(min(i) for i in list_keep.values()))

length_output_timeseries = 1
n_inputs = 1
n_outputs = 1
# length_test_file calculated with the size of the test file


testFiles = glob.glob("Bin_Files/test_*.bin")
print(testFiles)


###################################################################################################################
# List of models to test, test directories creation and test result files initialisation
###################################################################################################################

models = glob.glob("*/Models/*.h5")
testFolders = []
testResultFiles = []
listentryType = []
list_n_params = []

for model in models :
    pathList = model.split('\\')
    modelName = pathList[-1][:-3]
    testFolders.append(pathList[0]+'\\'+"test_"+modelName)
    
    # Get type of entry vector (fibonacci, linear, 1_out_10)
    entryType = pathList[0].split('_')[:-1]
    entryType = '_'.join(entryType)
    # we have FF_1_out_10_2layers_512 we want 1_out_10
    entryType = "_".join(entryType.split('_')[1:-2])
    print(entryType)
    listentryType.append(entryType)
    os.makedirs(testFolders[-1], exist_ok=True)
    testResultFile = testFolders[-1]+"\\test_results"+modelName+".csv"
    testResultFiles.append(testResultFile)
    file = open(testResultFile,"w")
    file.write("Tests;MSE;RMSE;MAE;SMAPE;R²\n")
    file.close()

for i in range(len(models)) :
    model_obj = tf.keras.models.load_model(models[i])
    list_n_params.append(model_obj.count_params())
    del model_obj

print("models : ", models)
with open("n_params.csv", "w") as file :
    file.write("Model;Number of parameters\n")
    for i in range(len(models)) :
        file.write(models[i]+";"+str(list_n_params[i])+"\n")

sys.exit()

    
globalTestResultFile = "test_results.csv"
file = open(globalTestResultFile,"w")
file.write("Tests;Models;MSE;RMSE;MAE;SMAPE;R²\n")
file.close()


for test_bin_path in testFiles :
    
    testName = test_bin_path.split('\\')[-1][5:-4]
    
    print("\n"+
          "###############################\n"+
          "Test File : "+ testName +"\n"+
          "###############################\n")
    
     
    file_array = np.fromfile(test_bin_path, dtype=np.float32)
    file_array= file_array.reshape((2,int(len(file_array)/2))).T
    # print(file_array.shape)
    
    # Get the length of the test file
    length_test_file = len(file_array)
    print('Length of the test file: ', length_test_file)

    
    list_points = create_list_possible_points([length_test_file],max_input_timeseries, length_output_timeseries)
    test_dataset = tf.data.Dataset.from_tensor_slices(list_points)
    test_dataset = test_dataset.batch(batch_size)
    
    # n_batch = length_test_file // batch_size - 1
    print('Number of batches: ', len(test_dataset), " (batch size: ", batch_size, ")")
    
    for m in range(len(models)) :
        print("\nModel : "+ models[m]+"\n")
        
        model = tf.keras.models.load_model(models[m], compile=False)
        # get number of parameters
        n_params = model.count_params()
        list_n_params.append(n_params)
        
        x_test = []
        y_test = []
        y_pred = []
        n_batch = 0
        
        t_test = time.time()
        
        for coord_batch in test_dataset :
            n_batch += 1
            coord_batch = coord_batch.numpy()
            batch_inputs, batch_outputs = create_batch(points_list=coord_batch, files_array=[file_array], length_files=[length_test_file], n_inputs=1, n_outputs=1,length_input_timeseries=max_input_timeseries, length_output_timeseries=length_output_timeseries, lines_keep=list_keep[listentryType[m]])
            x_test.append(batch_inputs[:,-1].copy())
            y_test.append(batch_outputs[:,-1].copy())
            y_pred.append(model.predict_on_batch(batch_inputs))

            if n_batch % 100 == 0:
                print("Batch: ", n_batch, "/", len(test_dataset), " - Time: ", time.time() - t_test, "s")

        print("Total time :",time.time() - t_test, "s")
        x_test = np.concatenate(x_test,axis = 0).reshape(-1,1)
        y_test = np.concatenate(y_test,axis = 0).reshape(-1,1)
        y_pred = np.concatenate(y_pred,axis = 0)
        
        ##########
        # Calcul indicateurs globaux
        ##########
        
        MSE   = (sum((y_test-y_pred)**2)/len(y_test))[0]
        RMSE  = math.sqrt(MSE)
        MAE   = (sum(abs(y_test-y_pred))/len(y_test))[0]
        
        y_     = np.mean(y_test)
        SAPE   = 2*abs(y_test-y_pred)/(abs(y_test)+abs(y_pred))
        SAPE_0 = SAPE[np.where((y_test < -0.01) | (y_test > 0.01))]
        SMAPE  = (100/len(SAPE_0)) * sum(SAPE_0)
        R2     = 1 - sum((y_test-y_pred)**2)[0] / sum((y_test-y_)**2)[0] 
        
        print("MSE   =", MSE)
        print("RMSE  =", RMSE)        
        print("MAE   =", MAE)
        print("SMAPE =", SMAPE,"%")
        print("R²    =", R2)
        
        file = open(testResultFiles[m],"a")
        file.write(testName + ";" + str(MSE).replace(".",",") + ";" + str(RMSE).replace(".",",") + ";" + str(MAE).replace(".",",") + ";" + str(SMAPE).replace(".",",") + "%" + ";" + str(R2).replace(".",",") + "\n")
        file.close()
        
        file = open(globalTestResultFile,"a")
        file.write(testName + ";" + models[m] + ";" + str(MSE).replace(".",",") + ";" + str(RMSE).replace(".",",") + ";" + str(MAE).replace(".",",") + ";" + str(SMAPE).replace(".",",") + "%" + ";" + str(R2).replace(".",",") + "\n")
        file.close()
        
        concat_output = np.concatenate((x_test, y_test, y_pred, SAPE), axis=1)
        filename_output = testFolders[m] + "\\" + testName + ".csv"
        
        


        np.savetxt(filename_output, concat_output, delimiter=';', newline='\n', header = "U_in;I_out_test;I_out_pred;SAPE")
        
        del x_test
        del y_test
        del y_pred
        del concat_output
        
        del batch_inputs
        del batch_outputs
        gc.collect()
        
        



sys.exit()












###################################################################################################################
# Load the model
###################################################################################################################
model_path = "./"
print("Loading model...")
print("list of models:", os.listdir(model_path))
list_models = [x for x in os.listdir(model_path) if x.endswith('.h5')]
list_models.sort()
last_model = model_path + "/" + list_models[-1]

print('Loading model: ', last_model)
model = tf.keras.models.load_model(last_model)
folder_output = "test_" + list_models[-1].split('.')[0]
if not os.path.exists(folder_output):
    os.mkdir(folder_output)
print('Model loaded')

###################################################################################################################
# Load the test data
###################################################################################################################

test_bin_path = '../Bin_Files/Test.bin'
test_file_obj = open(test_bin_path, 'rb')
# Get the length of the test file
size_test_file = os.stat(test_bin_path).st_size
length_test_file = int(size_test_file / (4 * (n_inputs + n_outputs)))
print('Length of the test file: ', length_test_file)
x_test = test_file_obj.read(length_test_file * 4 * n_inputs)
x_test = np.frombuffer(x_test, dtype=np.float32)
y_test = test_file_obj.read(length_test_file * 4 * n_outputs)
y_test = np.frombuffer(y_test, dtype=np.float32)

print(y_test)
y_pred = []


###################################################################################################################
# Test the model
###################################################################################################################

t_test = time.time()

n_batch = length_test_file // batch_size - 1
print('Number of batches: ', n_batch, " (batch size: ", batch_size, ")")

for batch in range(n_batch):
    batch_input = np.empty((batch_size, length_input_timeseries))
    batch_output = np.empty((batch_size, length_output_timeseries))
    for i in range(batch_size):
        batch_input[i] = x_test[i+batch*batch_size:i+batch*batch_size+length_input_timeseries]
        batch_output[i] = y_test[i+batch*batch_size+length_input_timeseries-length_output_timeseries:i+batch*batch_size+length_input_timeseries]
    pred_ = model.predict_on_batch(batch_input)
    y_pred.append(pred_)
    if batch % 10 == 0:
        print("Batch: ", batch, "/", n_batch, " - Time: ", time.time() - t_test, "s")
        

y_pred = np.array(y_pred)
y_pred = np.reshape(y_pred, np.product(y_pred.shape))



# Save prediction to csv (without the )
x_test_save = x_test[-len(y_pred):].T.reshape(-1,1)
y_test_save = y_test[-len(y_pred):].T.reshape(-1,1)
y_pred_save = y_pred.T.reshape(-1,1)
concat_output = np.concatenate((x_test_save, y_test_save, y_pred_save), axis=1)
filename_output = folder_output + "/output.csv"


np.savetxt(filename_output, concat_output, delimiter=',', newline='\n')
   


print(f"Test MSE: {np.mean(np.abs(y_pred_save-y_test_save))}")

plt.figure(figsize=(100,100))
plt.plot(y_pred[:100_000])
plt.plot(y_test[:100_000])
plt.show()