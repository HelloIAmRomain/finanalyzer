import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Activation, GRU, Dropout
import matplotlib.pyplot as plt
import time
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

###################################################################################################################
# Variables
###################################################################################################################

N_INPUTS = 1
N_OUTPUTS = 1
OUTPUT_SHAPE = 1
# batch_size = 1_024
BATCH_SIZE = 1024
EPOCHS = 10
INPUT_SHAPE = (128, N_INPUTS)
MODEL_TOPOLOGY = [128, 64, 32, 16, 8, 4, 2, 1]


def normalize_data(data):
    """
    Normalize the data between -1 and 1
    """
    data = data.astype(np.float32)
    norm_val = np.max(np.abs(data))
    data = data / norm_val
    return data, norm_val

def val_step(model, x_batch_val, y_batch_val, val_acc_metric) :
    val_logits = model([x_batch_val], training=False)
    val_acc_metric.update_state(y_batch_val, val_logits)

def smape(y_true, y_pred):
    """
    Symmetric mean absolute percentage error
    """
    return 100.0 * tf.reduce_mean(2.0 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred)))


def build_model(input_shape, output_shape, topology):
    """
    Build a GRU model for time series analysis
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for layer in topology :
        model.add( GRU( units=layer, # Dimensionality of the output space
                        # unroll=True, # Faster but needs more memory (set to False if there is memory issues)
                        return_sequences=True, # Whether to return the last output in the output sequence, or the full sequence
        ))
        model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    return model


def create_model_name(topology):
    """
    Create a list of the name of the models
    """
    name = "GRU_" + str(len(topology)) + "_layers_" + "_".join([str(layer) for layer in topology])
    return name


class GRURegressor:
    def __init__(self, input_shape=INPUT_SHAPE,
                 n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                 output_shape=OUTPUT_SHAPE, model_topology=MODEL_TOPOLOGY,
                 batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.normalization_value = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.model_topology = model_topology
        self._model = build_model(self.input_shape, self.output_shape, self.model_topology)
        self.model_loss = MeanSquaredError()
        self.model_optimizer = Adam()
        self.model_train_acc_metric = MeanSquaredError()
        self.model_val_acc_metric = MeanSquaredError()
        self.gru_path = "./gru"
        if not os.path.exists(self.gru_path):
            os.makedirs(self.gru_path)
        self.model_name = os.path.join(self.gru_path, create_model_name(self.model_topology))
        self.checkpoint_path = self.model_name + "/checkpoint.ckpt"
        self.history = None
        self.n_params = self._model.count_params()

    def __repr__(self):
        descrption = f"""
GRU Regressor
-------------
Input shape : {self.input_shape}
Output shape : {self.output_shape}
Model topology : {self.model_topology}
Parameters : {self.n_params}
Batch size : {self.batch_size}
Epochs : {self.epochs}
        """
        return descrption

    def save_model(self):
        """
        Save the model
        """
        self._model.save(self.model_name)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def train(self, timeseries, val_split=0.1, verbose=1, continue_training=False):
        """
        Train the model
        """
        t_init = time.time()
        if continue_training and os.path.exists(self.checkpoint_path):
            print(f"Loading model from {self.checkpoint_path}")
            self._model.load_weights(self.checkpoint_path)
        timeseries, norm_val = normalize_data(timeseries)
        self.normalization_value = norm_val
        val_index = int(len(timeseries) * (1 - val_split))
        train_dataset = timeseries[:val_index]
        val_dataset = timeseries[val_index:]
        # Transform the data into batches from timeseries
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.window(self.input_shape[0] + self.output_shape, shift=1, drop_remainder=True)
        train_dataset = train_dataset.flat_map(lambda window: window.batch(self.input_shape[0] + self.output_shape))
        train_dataset = train_dataset.shuffle(10_000)
        train_dataset = train_dataset.map(lambda window: (window[:-self.output_shape], window[-self.output_shape:]))
        train_dataset = train_dataset.batch(self.batch_size).prefetch(1)
        # TODO: Fix the batch creation
        print(train_dataset)
        # Transform the data into batches from timeseries (validation)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)
        val_dataset = val_dataset.window(self.input_shape[0] + self.output_shape, shift=1, drop_remainder=True)
        val_dataset = val_dataset.flat_map(lambda window: window.batch(self.input_shape[0] + self.output_shape))
        val_dataset = val_dataset.shuffle(10_000)
        val_dataset = val_dataset.map(lambda window: (window[:-self.output_shape], window[-self.output_shape:]))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(1)

        # Create the checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        verbose=verbose)
        # Compile the model
        self._model.compile(optimizer=self.model_optimizer, loss=self.model_loss)
        # Train the model
        self.history = self._model.fit(train_dataset,
                                      epochs=self.epochs,
                                      validation_data=val_dataset,
                                      callbacks=[checkpoint],
                                      verbose=verbose)
        # Save the model
        self.save_model()
        t_end = time.time()
        print(f"Training time : {t_end - t_init} seconds")

    def predict(self, x, load_model=False):
        """
        Predict with the model
        """
        t_init = time.time()
        # Load the model
        x = np.array(x) / self.normalization_value
        if load_model:
            self._model.load_weights(self.checkpoint_path)
        # Predict
        y_pred = self._model.predict(x) * self.normalization_value
        t_end = time.time()
        print(f"Prediction time : {t_end - t_init} seconds")
        return y_pred

    def evaluate(self, x, y, load_model=False):
        """
        Evaluate the model (SMAPE error)
        """
        t_init = time.time()
        x = np.array(x) / self.normalization_value
        y = np.array(y) / self.normalization_value
        # Load the model
        if load_model:
            self._model.load_weights(self.checkpoint_path)
        # Evaluate
        y_pred = self._model.predict(x) * self.normalization_value
        error = smape(y, y_pred)
        t_end = time.time()
        print(f"Evaluation time : {t_end - t_init} seconds")
        return error

    @property
    def summary(self):
        """
        Print the model summary
        """
        self._model.summary()

    def plot_model(self):
        """
        Plot the model
        """
        return tf.keras.utils.plot_model(self._model, show_shapes=True, dpi=96)

    def plot_history(self):
        """
        Plot the history of the model
        """
        if self.history is None:
            raise ValueError("The model has not been trained yet")
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()

    def pred_n_steps(self, timeseries, n_steps, load_model=False):
        """
        Predict n steps ahead
        """
        t_init = time.time()
        # Load the model
        timeseries, norm_val = normalize_data(timeseries)
        if load_model:
            self._model.load_weights(self.checkpoint_path)
        # Predict
        y_pred = []
        for i in range(n_steps):
            y_pred.append(self._model.predict(timeseries[:, i:i+self.input_shape[0]]) * norm_val)
            timeseries = np.append(timeseries[:, 1:], y_pred[-1], axis=1)
        t_end = time.time()
        print(f"Prediction time : {t_end - t_init} seconds")
        return np.array(y_pred)
