from sauron import Sauron
from convolution_nn import ConvolutionNN
import cv2
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADES = cv2.CascadeClassifier(os.path.join(PROJECT_ROOT, "cascades/data/haarcascade_frontalface_alt2.xml"))

if __name__ == "__main__":
    conv = ConvolutionNN(cv2, FACE_CASCADES, 500)
    conv.train()
    the_eye = Sauron(conv)


    while True:
        value = the_eye.recognize_rasp()
        print(value)