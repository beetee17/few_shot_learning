import keras
import tensorflow as tf

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16
 
from keras.layers import Flatten
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Concatenate, Dot, Lambda, Input, Dropout

from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2

from keras.optimizers import Adam
from keras import backend as K

import numpy as np

def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network

    model = Sequential()
    
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                    kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
                    kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                        kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
                        bias_initializer=initializers.RandomNormal(mean=0.5, stddev=0.01), 
                        kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
                        bias_initializer=initializers.RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01),
                        bias_initializer=initializers.RandomNormal(mean=0.5, stddev=0.01), kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.1),
                    bias_initializer=initializers.RandomNormal(mean=0.5, stddev=0.01)))
    

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer='zeros')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # return the model
    return siamese_net  
    
def get_pretrained_model(input_shape, num_dense=1, dense_size=(256)):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    pre_train = Sequential()
    base = VGG16(include_top=False, input_shape=input_shape)
  
    # freeze all layers but last 4
    for layer in base.layers[:-4]:
        layer.trainable = False

    for layer in base.layers:
        print(layer.name, layer.trainable)
    
    pre_train.add(base)
    pre_train.add(tf.keras.layers.Flatten())

    for i in range(num_dense):
        pre_train.add(Dense(dense_size[i]))
        pre_train.add(Dropout(0.5))
    
    for layer in pre_train.layers:
        print(layer.name, layer.trainable)

    embedding = Model(pre_train.input, pre_train.output, name="Embedding")
    
    

    # Generate the encodings (feature vectors) for the two images
    encoded_l = embedding(left_input)
    encoded_r = embedding(right_input)
    
    cos_sim = Dot(axes=1, normalize=True)([encoded_l, encoded_r])

     
    # Add a customized layer to compute the absolute difference between the encodings (1/diff to get similarity score)
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',kernel_initializer=initializers.Constant(value=5), bias_initializer='zeros')(cos_sim)

    # prediction = Lambda(lambda x: 1-x)(tanh)

    # # Connect the inputs with the outputs
    pretrained_model = Model(inputs=[left_input,right_input], outputs=prediction)

    # return the model
    return pretrained_model

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, val_inputs, val_labels, patience=0, batch_size=16):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.best_epoch = 0
        self.val_inputs = val_inputs
        self.val_labels = val_labels
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != 0:
            print('\n\n')

    def on_epoch_end(self, epoch, logs=None):
        
        print('\nEvaluating model on validations set')
        metrics = self.model.evaluate(x=self.val_inputs, y=self.val_labels, batch_size=self.batch_size, verbose=1, sample_weight=None, steps=None, return_dict = True)
        current = metrics['loss']

        # for plotting learning process
      
        logs['val_loss'] = metrics['loss']
        logs['val_acc'] = metrics['accuracy']

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch + 1
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch - epoch {}.".format(self.best_epoch))
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))
        else:
            self.model.set_weights(self.best_weights)
            print("set weights from epoch {}".format(self.best_epoch))
        