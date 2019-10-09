import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Filereader():
    def __init__(self, image_path = "pictures"):
        self.image_dir = os.path.join(BASE_DIR, image_path)

    # filters the photo dirs and name of the folder
    # NB: The dirs folder is the name of the  
    def retrive_file_names(self):
        paths = []
        directories = []
        for root, dirs, files in os.walk(self.image_dir):
           paths.append([os.path.join(root,file) for file in files if(file.endswith(".jpeg") or file.endswith(".png"))])
           directories.append([dir for dir in dirs])
        return (list(filter(lambda x: x != [], directories)), list(filter(lambda x: x != [], paths)))