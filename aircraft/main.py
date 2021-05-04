import sys
from typing import Optional

from fastapi import FastAPI

sys.path.append('c:\\Users\\Admin\\few_shot_learning\\')

from Utils import preprocess, testModels,  utils

import numpy as np
import os

import keras

### pip install fastapi
### pip install uvicorn[standard]
### cd C:\Users\Admin\few_shot_learning\aircraft
### uvicorn main:app --reload

print('loading model...')
model = keras.models.load_model(r'C:\Users\Admin\few_shot_learning\aircraft\models\vgg16_batch_50_norm.h5')

def get_pairs(path):

    support_set = list()

    query_img_dir = ['{}\query\{}'.format(path, i) for i in os.listdir(path + '\query')][0]

    support_imgs_dir = ['{}\support\{}'.format(path, i) for i in os.listdir(path + '\support')]
    
    query = plt.imread(query_img_dir)
    query = preprocess.pad_and_resize(query, desired_ratio=1.4, width=280, height=200)
    
    
    for img_dir in support_imgs_dir:
        support_img = plt.imread(img_dir)
        support_img = preprocess.pad_and_resize(support_img, desired_ratio=1.4, width=280, height=200)

        pair = [query, support_img]
        support_set.append(pair)

    
    return query_img_dir, support_imgs_dir, np.array(support_set)


app = FastAPI()


@app.get("/")
def read_root():

    return 'To get similarity scores between a query image and some support images, put them in a folder named "query" and "support" respectively. Then put the folders in another folder and pass the path to this root folder in the url like so:\nhttp://localhost:8000/get_predictions/?path=C:\\Users\\Admin\\few_shot_learning\\aircraft\\example'



@app.get("/get_predictions/")
def get_predictions(path: str):

    query_img_dir, support_imgs_dir, support_set = get_pairs(path)
    
    predictions = model.predict([support_set[:,0], support_set[:,1]])
    predictions = {support_imgs_dir[i] : str(predictions[i]) for i in range(len(support_imgs_dir))}


    print(predictions)

    return {"predictions": predictions}

