# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:21:42 2021

@author: thomas.framery
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow import keras
import time
import numpy as np
import gc
import glob

# Compiles a function into a callable TensorFlow graph. (deprecated arguments)
@tf.function
def parse_image(filename):
  global height
  global width
  # récupérer les infos de notre fichier
  parts = tf.strings.split(filename, os.sep)
  # récupération des labels de nos images
  label = parts[-2]
  label = [0,1] if label == 'pos' else [1,0]
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  # redéfinir le type de l'image
  image = tf.image.convert_image_dtype(image, tf.float32)/255
  # changement de taille des images 230 150
  image = tf.image.resize(image, [height, width])
  # image = tf.expand_dims(image, axis=0)
  # label = tf.expand_dims(label, axis=0)
  return image, label

def preprocessing(image, label):
    """
    augmentation de la bdd par des opérations de base comme
    les tourner, changer le contraste, décalage, zoom
    """
    # décalage des images
    image = tf.keras.preprocessing.image.random_shift(image, 0.02, 0.02)
    
    # ajouter du contraste à l'image
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # ajouter une rotation
    image = tf.keras.preprocessing.image.random_rotation(image.numpy(), 5, row_axis = 0, col_axis = 1, channel_axis = 2)

    # ajouter du zoom
    image = tf.keras.preprocessing.image.random_zoom(image,(0.8, 1.2), row_axis = 0, col_axis = 1, channel_axis = 2)
    # image = tf.keras.preprocessing.image.random_shear(image)
    
    # This op cuts a rectangular bounding box out of image. The top-left corner of the bounding box 
    # is at offset_height, offset_width in image, 
    # and the lower-right corner is at offset_height + target_height, offset_width + target_width
    image = tf.image.crop_to_bounding_box(image, int((height-t_height)/2), int((width-t_width)/2), target_height=t_height, target_width=t_width)
    # Linearly scales each image in image to have mean 0 and variance 1.
    image = tf.image.per_image_standardization(image)
    
    return image, label

@tf.function
def preprocessing_val(image, label):
    """
    Utiliser pour le passage de validation
    Sert à la préparation ds images
    """
    # crop une partie de l'image
    image = tf.image.crop_to_bounding_box(image,int((height-t_height)/2), int((width-t_width)/2), target_height=t_height, target_width=t_width)
    # Linearly scales each image in image to have mean 0 and variance 1.
    image = tf.image.per_image_standardization(image)
    return image, label

@tf.function
def train_step(x_batch_train,y_batch_train,model) :
    """
    passage du batch dans le modèle pour réaliser l'entrainement
    """
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, logits)
    # on récupère le(s) gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # on applique l'optimizer adam, avec le(s) gradients
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
    train_acc_metric.update_state(y_batch_train, logits)
        # on retourne notre loss
    return loss_value

@tf.function
def val_step(x_batch_val,y_batch_val,model) :
    """
    Similaire à train step
    """
    # on passe dans le model avec training false !!!
    val_logits = model(x_batch_val, training=False)      
    # on update les metric de validatio et non celle de training
    val_acc_metric.update_state(y_batch_val, val_logits)

if __name__ =="__main__":
    
    ###################################################################################################################
    #                                           Check GPU
    ###################################################################################################################
    
    # check si on a gpu, si oui on l'utilise pour les calculs 
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
        
    ###################################################################################################################
    #                                           VARIABLES
    ###################################################################################################################
    
    # définir les données d'entrée
    height = 150 #460 => 230      #MODIF
    width = 225 #150 => 150       #MODIF
    batch_size = 32
    epochs = 50
    # dimension pour crop les images
    t_height = 150 #420 => 210     #MODIF
    t_width = 225 #120             #MODIF
    
    # mettre à 3 pour RGB ou 1 pour NB
    inputShape = (t_height, t_width, 3)
    chanDim = -1
    plant = '150225'      #MODIF
    
      		# first CONV => RELU => CONV => RELU => POOL layer set
    inputs = keras.Input(shape=inputShape, name="digits")
    # Public API for tf.keras.applications namespace.
    model_incep = tf.keras.applications.InceptionV3(
        include_top=False, weights=None, input_tensor=inputs)
    
    ###################################################################################################################
    #                                           MODEL
    ###################################################################################################################
    
    x = model_incep.output
    x13 = Flatten()(x)
    x14 = Dense(254, activation="relu")(x13)      #MODIF
    # https://deeplizard.com/learn/video/dXB-KQYkzNU
    x15 = BatchNormalization()(x14)
    # https://www.youtube.com/watch?v=UcKPdAM8cnI
    x16 = Dropout(0.5)(x15)
      
      		# softmax classifier
    # 2 sortie 
    x17 = Dense(2, activation="relu")(x16)        #MODIF
    
    model = keras.Model(inputs=model_incep.input, outputs=x17)
    
    
    # définition de l'optimizer et du learning rate
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # on utilise la cross entropy
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True) 
    
    train_acc_metric = keras.metrics.BinaryAccuracy()
    val_acc_metric = keras.metrics.BinaryAccuracy()
    
    ###################################################################################################################
    #                                           RECUPERATION D'IMAGE
    ###################################################################################################################
    
    # on va chercher les photos désiré
    dataset = tf.data.Dataset.list_files("./*/*.png", shuffle=True)
    
    # on les transforme en glob pour faciliter le traitement
    # dataset = list_ds.map(parse_image)
    g = glob.glob("./*/*.png")
    nb_imagettes = int(len(g)*0.2)
    print('taille validation set : ',nb_imagettes)
    val_dataset = dataset.take(nb_imagettes) 
    #val_dataset = dataset.skip(35000).take(640) 
    val_dataset = val_dataset.batch(batch_size)
    
    train_dataset_first = dataset.skip(nb_imagettes)
    
    ###################################################################################################################
    #                                           PREPARER CHECKPOINT
    ###################################################################################################################

    # Vérifier que le dossier est bien créé
    checkpoint_path = os.path.abspath('./model') 
    # on récupère les caracs de notre modèle
    ckpt = tf.train.Checkpoint(model = model, optimizer = optimizer)
    # Manages multiple checkpoints by keeping some and deleting unneeded ones.
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt_incep3_150225")
    
    
    ###################################################################################################################
    #                                           REPRENDRE AVEC LA DERNIERE SAUVEGARDE
    ###################################################################################################################
    start_epoch = 0
    # checker s'il y a eu un enregistrement 
    if ckpt_manager.latest_checkpoint:
        # si oui on reprends les epochs de la ou on était
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) + 1
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('model restored')
    
    ###################################################################################################################
    #                                           DEBUT ENTRAINEMENT
    ###################################################################################################################
    
    # la boucle qui commence
    for epoch in range(start_epoch ,epochs):
        
        num_batch = 0
        # on tambouille/mélange tout
        train_shuffle = train_dataset_first.shuffle(800000)
        
        train_dataset = train_shuffle.batch(batch_size)
        
    
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
    
    
        # Iterate over the batches of the dataset.
        i = 0
        for batch_dataset in train_dataset :
            batch_dataset_data = tf.data.Dataset.from_tensor_slices(list(batch_dataset))
            batch_dataset = batch_dataset_data.map(parse_image, num_parallel_calls=16)
            batch_dataset = batch_dataset.map(lambda x,y : tf.py_function(preprocessing, [x,y], [tf.float32,tf.int32]),
                                              num_parallel_calls=16)
             
            a = np.asarray(list(batch_dataset.as_numpy_iterator()), dtype=object)
            y_batch_train = np.stack(a[:,1].ravel())
            x_batch_train = np.stack(a[:,0].ravel())
    
            
            loss_value = train_step(x_batch_train, y_batch_train, model)
            
        
                # Log every 200 batches.
            i += 1
            if i % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (i, float(loss_value))
                )
                print("Seen so far: %d samples" % ((i) * batch_size))
                
            del(x_batch_train)
            del(y_batch_train)
            del(a)
            del(loss_value)
            del(batch_dataset)
            del(batch_dataset_data)
            gc.collect()
    
        # Display metrics at the end of each epoch.
        # accuracy de chaque epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
    
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
    
        # Run a validation loop at the end of each epoch.
        cpt = 0
        
        for batch_dataset in val_dataset :
            batch_dataset_data = tf.data.Dataset.from_tensor_slices(list(batch_dataset))
            batch_dataset = batch_dataset_data.map(parse_image, num_parallel_calls=16)
            # batch_dataset = batch_dataset.map(lambda x,y : tf.py_function(preprocessing_val, [x,y], [tf.float32,tf.int32]),
            #                                   num_parallel_calls=16)
            batch_dataset = batch_dataset.map(preprocessing_val, num_parallel_calls=16)
            a = np.asarray(list(batch_dataset.as_numpy_iterator()), dtype=object)
    
            y_batch_val = np.stack(a[:,1].ravel())
            x_batch_val = np.stack(a[:,0].ravel())
            
            val_step(x_batch_val, y_batch_val, model)        
            
            cpt += 1
            if cpt % 20 == 0:
                print("Seen so far: %d samples" % ((cpt) * batch_size))
                #print('Last pred : {}'.format(val_logits))
                
            del(x_batch_val)
            del(y_batch_val)
            del(a)
            del(batch_dataset)
            del(batch_dataset_data)
            gc.collect()
                
        # on recupére les metrics dans une nouvelle variable
        val_acc = val_acc_metric.result()
        # pour print
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2f min" % ((time.time() - start_time)/60))
        
        
        val_acc_metric.reset_states()
        
        if val_acc >= 0.90 :
            # addresse où on enregistre le model eb format tf 
            model.save("./model/inceptionv3_"+plant+"_"+str(epoch+1)+"_epochs_"+str(float(val_acc))[:5]+".h5")
            ckpt.save(file_prefix=checkpoint_prefix)
            #ckpt_manager.save()
            
            #model.save_weights("minivgg_"+str(epoch+1)+"_epochs_"+str(float(val_acc))[:4]+"_2.h5")
    
    #         #tf.keras.models.save_model(model, "minivgg_"+str(epoch+1)+"_epochs_"+str(float(val_acc))[:4]+"_3.h5")
      
