import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Filereader():
    def __init__(self, image_path = "pictures"):
        self.image_dir = os.path.join(BASE_DIR, image_path)

    def retrive_file_names(self):
        paths = []
        directories = []
        for root, dirs, files in os.walk(self.image_dir):
           paths.append([file for file in files if(file.endswith(".jpg") or file.endswith(".png"))])
           directories.append(dirs)
        print(directories)
        print(paths)
                    
                    
file_reader = Filereader()

value = file_reader.retrive_file_names()
