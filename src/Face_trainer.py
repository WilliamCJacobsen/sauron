from Filereader import *
import numpy as np
from PIL import Image

class Face_trainer:
    def __init__(self, cv, cascade, picture_folder = "pictures"):
        self.filereader = Filereader(picture_folder)
        self.cascade = cascade
        self.folder = picture_folder
        self.cv = cv

    def train(self):
        (directories, files) = self.filereader.retrive_file_names()
        x_train = []
        labels = []

        face_cascade = self.cv.Cascade
        pil_image = Image.open(path).convert("L")
        pil_image_array = np.array(pil_image, 'uint8')
        faces = self.cascade.detectMultiScale(pil_image_array, scaleFactor=1.5, minNeighbors=5)

        for (x,y,w,h) in faces:
            roi = pil_image_array[x+w, y+h]
            x_train.append(roi)

ft = Face_trainer(1,2)

ft.print()