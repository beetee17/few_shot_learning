import os
import numpy as np
import h5py

def save_data(path='', **kwargs):
    # save it once to allow load from file

    # train_pairs, train_labels and X_test all contain 0s and 1s which can be converted to unsigned 8 bit ints
    # use np.astype(np.uint8) to convert array before saving

    print('saving data...')
    for key, value in kwargs.items():

        with h5py.File('{}{}.h5'.format(path, key), 'w') as hf:

           if type(value).__module__ == np.__name__:

                # train_pairs and X_test are image sets which are just 0s and 1s
                # train_labels also contain only 0s and 1s which 
                # can be converted to unsigned 8 bit ints
                # use np.astype(np.uint8) to convert array before saving

                value = value.astype(np.uint8)
                hf.create_dataset(key, value.shape, h5py.h5t.STD_U8BE, data = value)
                
                print('saved {} to /{}{}.h5!'.format(key, path, key))

           else:

                # If data is a list of strings
                dt = h5py.special_dtype(vlen=str) 
                value = np.array(value, dtype=dt) 
                hf.create_dataset(key, data = value)

                print('saved {} to /{}{}.h5!'.format(key, path, key))
            
    
    return None

#load from file
def load_data(path='', **kwargs):


    all_data = {}

    for key, value in kwargs.items():
     
        if value:

            with h5py.File('{}{}.h5'.format(path, key), 'r') as hf:

                if type(hf[key][0]) == bytes:
             
                    data = [item.decode('utf8') for item in hf[key][:]] # convert from bytes to str
                else:
                    data = hf[key][:]

                all_data.update({key : data}) 
                print('{} loaded from /{}{}.h5!'.format(key, path, key))

    return all_data
