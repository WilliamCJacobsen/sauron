from Filereader import Filereader 
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf

class Face_trainer:
    def __init__(self, cv, cascade, image_size:int, picture_folder = "pictures"):
        self.image_size = image_size
        self.filereader = Filereader(picture_folder)
        self.cascade = cascade
        self.folder = picture_folder
        self.cv = cv
    
    def get_names(self):
        (names, _) = self.filereader.retrive_file_names()
        return names
    
    def train_label(self):
        (directories, files) = self.filereader.retrive_file_names()
        x_train = []
        labels = []
        for (index, face_dir) in enumerate(files):
            counter = 0
            counter_image = 0
            for face in face_dir:
                pil_image = Image.open(face).convert("L")
                pil_image_array = np.array(pil_image, 'uint8')
                faces = self.cascade.detectMultiScale(pil_image_array, scaleFactor=1.5, minNeighbors=5)
                counter_image+= 1
                for (x,y,w,h) in faces:
                    roi = pil_image_array[x:x+w, y:y+h]
                    if roi.size != 0:
                        counter += 1
                        roi = tf.keras.utils.normalize(roi,axis=-1,order=2)
                        roi = self.cv.resize(roi, dsize=(self.image_size,self.image_size), interpolation=self.cv.INTER_NEAREST) 
                        x_train.append(roi)
                        labels.append(index)
            print(f"directory: {directories[index]}, there are {counter} amount of faces out of {counter_image} images!")
        return (x_train, labels)

    def train(self):
        (x_train, labels) = self.train_label()
        recognizer =  self.cv.face.LBPHFaceRecognizer_create()
        recognizer.train(x_train, np.array(labels))
        recognizer.save("trainer.yml")
