import numpy as np
import os
import random

from matplotlib import pyplot as plt

from PIL import Image
import math
from itertools import combinations
from sklearn.utils import shuffle


def pad_img(img, desired_ratio):

    try:
        img = Image.fromarray(img)
    except TypeError:
        pass

    width, height = img.size
 
    aspect_ratio = width / height

    if aspect_ratio > desired_ratio:
        # image is too 'long' -> pad the top and bottom
        required_height = int(width // desired_ratio)
        required_padding = (required_height - height) // 2
        print(height, width, required_height)

        new_img = Image.new("RGB", (width, required_height))  
        new_img.paste(img, (0, required_padding))


    elif aspect_ratio < desired_ratio:
        # image is too 'tall' -> pad the sides
        required_width = int(height * desired_ratio)
        print(height, width, required_width)
        required_padding = int((required_width - width) // 2)
        
        new_img = Image.new("RGB", (required_width, height))  
        new_img.paste(img, (required_padding, 0))


    else:
        # image is perfect ratio -> no padding required
        return img

    return new_img

def pad_and_resize(img, desired_ratio, width, height):

    padded_img = pad_img(img, desired_ratio)
    
    result = padded_img.resize((width, height))

    return np.array(result)


def get_all_X_Y(all_imgs_dir, desired_ratio, width, height):
    
    X = np.array([])
    Y = []
    count = 0

    for label, fns in all_imgs_dir.items():
        for fn in fns:
            img = np.array(Image.open(fn).convert("RGB"))
            new_img = pad_and_resize(img, desired_ratio, width, height)
        
            if X.size == 0:
                X = np.array([new_img])
                Y.append(label)
                continue

            try:
                X = np.vstack((X, [new_img]))
                Y.append(label)  

            except Exception as e:
                print(e)
                print(fn)
            
            if count % 200 == 0:
                print(X.shape, len(Y))

            count += 1
    
    return X.astype("int"), Y

def make_positive_pairs(X, Y, classes, N):

    # this function will produce N pairs of positive samples

    pairs = []
    labels = []
    raw_labels = []

    for label in classes:
        print(label)
        pos_i = [i for i, y in enumerate(Y) if y == label]
        comb = combinations(pos_i, 2)
        for c in comb:
            i = c[0]
            j = c[1]
            x1 = X[i]
            x2 = X[j]
            pairs += [[x1, x2]]
            labels += [1]
            raw_labels += [[Y[i], Y[j]]]

    pairs, labels, raw_labels = shuffle(pairs, labels, raw_labels)
        
    return np.array(pairs[:N]), np.array(labels[:N]), raw_labels[:N]

def make_negative_pairs(X, Y, N):

    # this function will produce N pairs of positive samples

    pairs = []
    labels = []
    raw_labels = []

    pairs_per_img = math.ceil(N/len(X))

    for i in range(len(X)):
        if i % 100 == 0:
                print(i)
        for k in range(pairs_per_img):
                        
            x1 = X[i]
            class_ = Y[i]

            # get non-match
        
            J = [j for j, y in enumerate(Y) if y != class_]

            j = np.random.choice(J)
        
            x2 = X[j]

            pairs += [[x1, x2]]
            labels += [0]
            raw_labels += [[Y[i], Y[j]]]
        
    pairs, labels, raw_labels = shuffle(pairs, labels, raw_labels)
        
    return np.array(pairs), np.array(labels), raw_labels