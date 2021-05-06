import sys
sys.path.append('c:\\Users\\Admin\\few_shot_learning\\')
from Utils import preprocess, testModels,  utils

import io
import os
import shutil
import requests
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional, List
from keras.models import load_model


from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, File, UploadFile



### pip install fastapi
### pip install uvicorn[standard]
### cd C:\Users\Admin\few_shot_learning\aircraft\fastapi
### uvicorn main:app --reload-dir C:\Users\Admin\few_shot_learning\aircraft\fastapi\main.py

print('loading model...')
model = load_model(r'C:\Users\Admin\few_shot_learning\aircraft\models\vgg16_batch_50_norm.h5')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
db = []
templates = Jinja2Templates(directory='templates')



@app.post("/get_predictions/")
async def create_upload_files(request : Request, files: List[UploadFile] = File(...)):
    """Receives a list of files uploaded by the user via the form in root webpage. Saves the files into the static folder and also gets the model's predictions for each support/query pair. Returns a HTML Response with the prediction outputs and the filenames for visualisation purposes"""

    support_set = list()
    queries = list()
    all_pairs = list()
    filenames = list()

    # for each file: read and write into the static/images dir
    # filter the files into support and query images by looking at the filenames
    # pad and resize the images to allow input into our model (280x200)px
    # feed the images into the model and return the predictions as a list
    for file in files:
     
        contents = await file.read()

        file_name = os.getcwd() + '\\static\\images\\' + file.filename.replace('/', '\\')
        filenames.append(file_name.replace(os.getcwd() + '\\static\\', ''))
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)

 
        image_bytes = Image.open(io.BytesIO(contents))
        
        image = np.array(image_bytes)

        image = preprocess.pad_and_resize(image, desired_ratio=1.4, width=280, height=200, crop_bottom=20)

        image_bytes = Image.fromarray(image)
        image_bytes.save(file_name)

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

    
    return  templates.TemplateResponse('predictions.html', {'request' :  request, 'filenames' : filenames, 'predictions' : predictions})


@app.get("/")
def read_root(request : Request):

    # ensure the static/images dir is cleared before user uploads any files 
    try:

        path = 'C:\\Users\\Admin\\few_shot_learning\\aircraft\\fastapi\\static\\images'

        for dir_ in os.listdir(path):

            shutil.rmtree(path + '\\' + dir_)

    except Exception as e:

        print(e)

    return templates.TemplateResponse('home.html', {'request' : request})
    


