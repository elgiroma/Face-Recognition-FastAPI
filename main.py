from fastapi import FastAPI, File, UploadFile, Form
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from os.path import join as joinpath
from fastapi import FastAPI, File, UploadFile, Form, status
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
import shutil
import PIL
import cv2
import numpy as np
import base64

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

app = FastAPI()

@app.get('/')
async def root():
    return {"message": "Bienvenido"}

@app.post('/file')
async def _file_upload(
        image: UploadFile = File(...),
        data: UploadFile = File(...),
):

    file_location = f"image_stored/123.jpg"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(image.file, file_object) 

    imagefile = cv2.imread('image_stored/123.jpg')

    load_data = torch.load(data.file)
    embedding = load_data[0]
    name_list = load_data[1]

    img = Image.fromarray(imagefile)
        #print(img)
    img_cropped, prob = mtcnn(img, return_prob=True) 
        
        
    if img_cropped is not None:               
        if prob>0.90:
            emb = resnet(img_cropped[0].unsqueeze(0)).detach() 
                        
            dist_list = [] # list of matched distances, minimum distance is used to identify the person
                        
            for idx, emb_db in enumerate(embedding):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            min_dist = min(dist_list) # get minumum dist value
            min_dist_idx = dist_list.index(min_dist) # get minumum dist index
            name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                        
            dictionary = { 
                    "Nombre": name,
                    "Cercania": min_dist,
                    "Mensaje": "Mientras mas cerca del cero, mas se parece a " + name + " en la base de datos proporcionada"
                }            
                        
            if min_dist<0.90:
                return dictionary
            else:
                print("No match")
                return "No se encontró un match en la base de datos"
        else:
            print("Probabilidad muy baja de que sea una cara")
            return "No se detecto una cara"
    else:
        print("No se detecto una cara")
        return "No se detecto una cara"

    