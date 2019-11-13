from io import BytesIO
from picamera import PiCamera
import cv2


class Sauron:
    def __init__(self, convolution_nn):
        self.convolution_nn = convolution_nn
        self.cam = PiCamera()
        self.cap = cv2.VideoCapture(0)

    def recognize_rasp(self):
        frame = BytesIO()
        self.cam.capture(frame, 'png')
        value = conv.recoginze(cv2.imdecode(frame.getvalue(), 0))
        return value

    def recoginze(self):
        ret, frame = self.cap.read()
        value = self.convolution_nn.recoginze(frame)
        return value