import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from Utils.Class import Predictor, FSL, Random, Nearest_Neighbour, Model_Nearest_Neighbour
from Utils.saveLoad import save_data, load_data
from Utils import preprocess, testModels, buildModel, utils

import numpy as np
import random

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf

from keras.callbacks import History 

from keras import backend as K


if __name__ == '__main__':

    purpose = input("What would you like to do?\n1. Train\n2. Test\n").strip()

    if purpose == str(1):
        
        ## TRAIN A MODEL

        # Load training and validation datasets
    
        while True:

            path = input("Enter the absolute path to the directory containing your training and validation .h5 files:\n").strip()

            try:
                kwargs = {'train_pairs' : True,
                    'train_labels' : True,
                    'val_pairs' : True,
                    'val_labels' : True}

                print('loading training and validation set...\n')
                data = load_data(path=path, **kwargs)

                train_labels = data['train_labels']
                val_labels = data['val_labels']
                val_inputs = [data['val_pairs'][:,0], data['val_pairs'][:,1]]
                train_inputs = [data['train_pairs'][:,0], data['train_pairs'][:,1]]

                break
            
            except Exception as e:
                print(e)
                continue

        del data
        import gc
        gc.collect()

        load = input('Would you like to load an existing model? (y/n)\n').strip()
        
        while load != 'y' and load != 'n':
            
            print('Enter "y" or "n" only')
            load = input('Would you like to load an existing model? (y/n)\n').strip()
            
        if load == 'y':

            # User wants to continue training a saved model

            while True:

                try:

                    model_fn = input('Enter the path to the model:\n').strip()
                    model = keras.models.load_model(model_fn)
                    break

                except Exception as e:

                    print(e)
                    continue


        elif load == 'n':

            # train a new model from scratch

            while True:

                try:

                    dense_layers = list(map(int, input("Enter the size of each of the model's dense layers (separated by commas)\n").strip().split(',')))

                    assert min(dense_layers) >= 0, 'Please enter a non-negative number'
                    break

                except Exception as e:
                    print(e)
                    continue
            
            while True:

                try:

                    model = input('Enter the pretrained model you would like to fine-tune ("vgg16", "xception" "resnet"):\n').strip()

                    assert model == "vgg16" or model == "xception" or model == "resnet", 'Please enter a valid model name'
                    break

                except Exception as e:
                    print(e)
                    continue 
            
            
            while True:

                try:

                    num_layers = int(input('Enter the number of layers to fine-tune:\n').strip())

                    assert num_layers >= 0, 'Please enter a non-negative number'
                    break

                except Exception as e:
                    print(e)
                    continue

            while True:

                try:

                    input_shape = list(map(int, input("Enter the height and width of your data (separated by a comma):\n").strip().split(',')))

                    input_shape.append(3)

                    assert len(input_shape) == 3, 'Please enter a width and a height'
                    assert min(input_shape) >= 0, 'Please enter a non-negative number'
                    break

                except Exception as e:
                    print(e)
                    continue
        
            while True:
                
                try:

                    batch_norm = input('Would you like to use batch normalisation in the network? (y/n)\n').strip()

                    assert batch_norm == 'y' or batch_norm == 'n', 'Enter "y" or "n" only'

                    to_bool = {'y' : True, 'n' : False}

                    break

                except Exception as e:
                        print(e)
                        continue 

            while True:

                try:

                    dropout = float(input('Enter the dropout percentage as a decimal, or enter 0 if no dropout is to be used:\n').strip())

                    assert dropout >= 0, 'Please enter a non-negative number'
                    break

                except Exception as e:
                    print(e)
                    continue
            
            print('\nBuilding model...\n')
            
            model = buildModel.get_pretrained_model(input_shape=input_shape, base=model, trainable=num_layers, num_dense=len(dense_layers), dense_size=dense_layers, batch_norm=to_bool[batch_norm], dropout=dropout)

        summary = input("Would you like to see a summary of your siamese network? (y/n)\n").strip()

        while summary != 'y' and summary != 'n':
            
            # View summary of model's architecture

            print('Enter "y" or "n" only')
            summary = input("Would you like to see a summary of your model? (y/n)\n").strip()

        if summary =='y':

            model.summary()


        # Specify the details of the model's training e.g. optimizer's learning rate, early stopping parameters, # of epochs to train, training batch size

        while True:

            try:
        
                learning_rate = float(input("Enter a learning rate for the model\n").strip())
                optimizer = keras.optimizers.adam_v2.Adam(learning_rate = learning_rate)

                print('\nCompiling model...')
                model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

                print('set learning rate as {}'.format(model.optimizer.learning_rate))
                break

            except ValueError as e:
                print(e)
                print('Please enter a number')
                continue
        
        # for plotting of model's learning curve post-training
        hist = History()

        while True:

            try:

                patience = input("Enter early stopping tolerance level\n").strip()

                early_stop = buildModel.EarlyStoppingAtMinLoss(patience=int(patience), val_inputs=val_inputs, val_labels=val_labels)
                break

            except ValueError:
                
                print('Please enter a number')
                continue

        while True:

            try:

                batch_size = int(input("Enter training batch size\n").strip())
                epochs = int(input("Enter # of epochs to train\n").strip())
                break

            except ValueError:
                
                print('Please enter a number')
                continue
        
        print("Starting training...")
        model.fit(train_inputs, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[early_stop, hist])


        save = input('Would you like to save this model? (y/n)\n')

        while save != 'y' and save != 'n':
            
            print('Enter "y" or "n" only')
            save = input('Would you like to save this model? (y/n)\n')

        if save =='y':

            while True:

                try:

                    save_fn = input('Enter the path you would like to save this mode to:\n')
                    keras.models.save_model(model, save_fn)
                    print('saved to {}!'.format(save_fn))
                    break
                    
                except Exception as e:

                    print(e)
                    continue

    elif purpose == str(2):

        # User wants to test a saved model against some benchmark model(s), if any
        
        models = []

      
        while True:

            choice = input('Add a model to evaluate\n1: Saved Model\n2: Model Nearest Neighbour\n3: Naive Nearest Neighbour\n4: Random Guess\n5: Done\n').strip()

            # Add the specified model 
            if choice == str(1):

                while True:

                    try:

                        model_fn = input('Enter the path to the model:\n').strip()
                        model = tf.keras.models.load_model(model_fn)
                        break

                    except Exception as e:

                        print(e)
                        continue

                name = input('Name this model:\n').strip()
                models.append(FSL(model, name=name))

            elif choice == str(2):

                models.append(Model_Nearest_Neighbour())
            
            elif choice == str(3):

                models.append(Nearest_Neighbour())
            
            elif choice == str(4):

                models.append(Random())
            
            elif choice == str(5):
                break

            print('Your current models:\n{}'.format(models))

        data = load_data(path="D://", **{'X_test' : True, 'Y_test' : True})

        # Specify the testing parameters
        a = int(input('Enter lower bound of ways to test the models:\n').strip())
        b = int(input('Enter upper bound of ways to test the models:\n').strip())
        N = range(a, b+1)

        NUM_TEST = int(input('Enter number of tests per model:\n').strip())

        acc = testModels.get_accuracy(models, data['X_test'], data['Y_test'], N, NUM_TEST)

        save = input('Would you like to save the accuracies? (y/n)\n')

        while save != 'y' and save != 'n':
        
            print('Enter "y" or "n" only')
            plot = input('Would you like to save the accuracies? (y/n)\n')

        # plot the test accuracies for each model
        if save == 'y':

            while True:

                try:

                    save_fn = input('Enter the path to the save file:\n').strip()
                    testModels.plot_accuracy(acc, save=save_fn)
                    break

                except Exception as e:

                    print(e)
                    continue
        else:

            testModels.plot_accuracy(acc)
        