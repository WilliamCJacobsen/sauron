from sauron import Sauron
from convolution_nn import ConvolutionNN
import cv2
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADES = cv2.CascadeClassifier(os.path.join(PROJECT_ROOT, "cascades/data/haarcascade_frontalface_alt2.xml"))

if __name__ == "__main__":
    conv = ConvolutionNN(cv2, FACE_CASCADES, 200)
#    conv.model_summary()
    conv.training()

    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        value = conv.recoginze(frame)
        print(value)