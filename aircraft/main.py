import sys
from typing import Optional

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

@app.post("/uploadfiles/")
async def create_upload_files(request : Request, files: List[UploadFile] = File(...)):

    for file in files:
        contents = await file.read()

        db.append(contents)

    return {'filenames' : [file.filename for file in files]}

@app.get("/")
def read_root(request : Request):
    
    return templates.TemplateResponse('home.html', {'request' : request,
                                                    })
    


# http://localhost:8000/get_predictions/?path=path\\to\\folder
@app.get("/get_predictions/")
def get_predictions(request : Request):
    response = []
    for f in db:
        response.append(Response(content=f))
    print(response)
    return templates.TemplateResponse('predictions.html', {'request' : request, 'files' : response})
    # if the dir exists in the current working directory
    if os.path.isdir(os.getcwd() + '\\' + path):

        path = os.getcwd() + '\\' + path
        
    # if the dir is not a full path and not in cwd, check if it exists
    elif len(path.split('\\')) == 1:
        for abs_path, directories, files in os.walk(r'C:\Users\Admin'):
            
            if path in directories:
                path = os.path.join(abs_path, path)
                print('found %s' % os.path.join(abs_path, path))

    # the dir is not in cwd and was not found in local search -> it may be a full path
            
    query_img_dir, support_imgs_dir, support_set = get_pairs(path)
    
    predictions = list(model.predict([support_set[:,0], support_set[:,1]]).flatten())
    # predictions = {support_imgs_dir[i].replace(path + '\\support\\', '') : str(predictions[i]) for i in range(len(support_imgs_dir))}


    print(predictions)

    
    return templates.TemplateResponse('predictions.html', {'request' : request,
                                                           'query_img_dir' : query_img_dir,
                                                           'support_imgs_dir' : support_imgs_dir,
                                                           'predictions' : predictions}) 

    return {"predictions": predictions}



