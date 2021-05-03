import keras
import tensorflow as tf

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
 
from keras.layers import Flatten
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Concatenate, Dot, Lambda, Input, Dropout, BatchNormalization

from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import constraints

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

def get_feature_extractor(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    base = VGG16(include_top=False, input_shape=input_shape)
    base.add(tf.keras.layers.Flatten())

    extractor = Model(base.input, base.output, name="Feature Extractor")

    encoded_l = extractor(left_input)
    encoded_r = extractor(right_input)
    
    cos_sim = Dot(axes=1, normalize=True)([encoded_l, encoded_r])

    model = Model(inputs=[left_input,right_input], outputs=cos_sim)

    return model


def get_pretrained_model(input_shape, num_dense=1, dense_size=(256), base='vgg16', trainable=1, batch_norm=True, dropout=0):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    pre_train = Sequential()
        
    if base == 'resnet':
        base = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
        pre_train.add(base)

        for layer in base.layers[:-trainable]:
            layer.trainable = False
            print(layer.name, layer.trainable)

        for layer in base.layers[-trainable:]:
            layer.trainable = True
            print(layer.name, layer.trainable)
        

    elif base == 'vgg16':
        base = VGG16(include_top=False, input_shape=input_shape)
  
        # freeze all layers but last few, specified by trainable kwarg
        for layer in base.layers[:-trainable]:
            layer.trainable = False
            pre_train.add(layer)

        for layer in base.layers[-trainable:]:
            if batch_norm:
                pre_train.add(BatchNormalization())
            layer.trainable = True
            pre_train.add(layer)


    pre_train.add(tf.keras.layers.Flatten())

    for i in range(num_dense):
        if batch_norm:
            pre_train.add(BatchNormalization())
        pre_train.add(Dense(dense_size[i]))
        # pre_train.add(Dense(dense_size[i], kernel_constraint=constraints.max_norm(2.)))
        if dropout > 0:
            pre_train.add(Dropout(dropout))

    embedding = Model(pre_train.input, pre_train.output, name="Embedding")
    

    # Generate the encodings (feature vectors) for the two images
    encoded_l = embedding(left_input)
    encoded_r = embedding(right_input)
    
    cos_sim = Dot(axes=1, normalize=True)([encoded_l, encoded_r])

     
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',kernel_initializer=initializers.Constant(value=5), bias_initializer='zeros')(cos_sim)

    # prediction = Lambda(lambda x: 1-x)(tanh)

    # # Connect the inputs with the outputs
    pretrained_model = Model(inputs=[left_input,right_input], outputs=prediction)

    # return the model
    return pretrained_model 

