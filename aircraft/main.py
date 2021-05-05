import sys
from typing import Optional
import requests
from typing import List
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

sys.path.append('c:\\Users\\Admin\\few_shot_learning\\')

from Utils import preprocess, testModels,  utils

import matplotlib.pyplot as plt
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
db = []
templates = Jinja2Templates(directory='templates')
from PIL import Image
import io


@app.post("/uploadfiles/")
async def create_upload_files(request : Request, files: List[UploadFile] = File(...)):

    support_set = list()
    queries = list()
    all_pairs = list()

    for file in files:
     
        contents = await file.read()
        
        image = Image.open(io.BytesIO(contents))
        
        image = np.array(image)

        image = preprocess.pad_and_resize(image, desired_ratio=1.4, width=280, height=200)

        if 'support' in file.filename:
            support_set.append(image)

        elif 'query' in file.filename:
            queries.append(image)

    for query_img in queries:
        
        all_pairs.append([[query_img, support_img] for support_img in support_set])
    
    all_pairs = np.array(all_pairs)

    predictions = list()
 
    for query_pairs in all_pairs:

        prediction = model.predict([query_pairs[:,0], query_pairs[:,1]]).flatten()
        predictions.append(list(map(float, prediction)))


    return {'filenames' : [file.filename for file in files], 'predictions' : predictions}

@app.get("/")
def read_root(request : Request):
    
    return templates.TemplateResponse('home.html', {'request' : request,
                                                    })
    




