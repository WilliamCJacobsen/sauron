from sauron import Sauron
from convolution_nn import ConvolutionNN
import cv2
import os
import time
from io import BytesIO
from picamera import PiCamera

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADES = cv2.CascadeClassifier(os.path.join(PROJECT_ROOT, "cascades/data/haarcascade_frontalface_alt2.xml"))

if __name__ == "__main__":
    conv = ConvolutionNN(cv2, FACE_CASCADES, 200)
#    conv.model_summary()
    conv.training()
    cam = PiCamera()

    while True:
        frame = BytesIO()
        cam.capture(frame, 'png')
        value = conv.recoginze(cv2.imdecode(frame.getvalue(), 0))
        print(value)
