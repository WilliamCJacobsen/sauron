from Filereader import *
import numpy as np
from PIL import Image


class Face_trainer:
    def __init__(self, cv, cascade, picture_folder = "pictures"):
        self.filereader = Filereader(picture_folder)
        self.cascade = cascade
        self.folder = picture_folder
        self.cv = cv
        self.names = {}

    def train_label(self):
        (directories, files) = self.filereader.retrive_file_names()
        x_train = []
        labels = []

        for (index, face_dir) in enumerate(files):
            for face in face_dir:
                pil_image = Image.open(face).convert("L")
                pil_image_array = np.array(pil_image, 'uint8')
                print(pil_image_array)
                faces = self.cascade.detectMultiScale(pil_image_array, scaleFactor=1.5, minNeighbors=5)
                for (x,y,w,h) in faces:
                    roi = pil_image_array[x:x+w, y:y+h]
                    x_train.append(roi)
                    labels.append(index)
        return (x_train, labels)

    def train(self):
        (x_train, labels) = self.train_label()
        recognizer =  self.cv.face.LBPHFaceRecognizer_create()
        recognizer.train(x_train, np.array(labels))
        recognizer.save("trainer.yml")
