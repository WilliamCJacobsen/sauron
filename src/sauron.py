from io import BytesIO
from picamera import PiCamera
import cv2
import numpy as np

# set to true if captures are too slow
# the video port bypasses some postprocessing, sacrificing quality for framerate
USE_VIDEO_PORT = False
RESOLUTION = (1296, 972) # (width, height). use a 4:3 resolution for max FOV

class Sauron:
    def __init__(self, convolution_nn):

        def ceil_mul(self, x, mul):
            mod = x % mul
            return x if mod == 0 else x + mul - mod

        shape = (ceil_mul(RESOLUTION[1], 16), ceil_mul(RESOLUTION[0], 32), 3)

        self.convolution_nn = convolution_nn
#        self.cam = PiCamera()
        self.cap = cv2.VideoCapture(0)
        self.cam = PiCamera(resolution=RESOLUTION)
        self.buf = np.empty(shape, dtype=np.uint8)



    def recoginze(self):
        ret, frame = self.cap.read()
        value = self.convolution_nn.recoginze(frame)
        if value:
            print(value)
        return value


    def recognize_pi(self):
        cam.capture(self.buf, format="bgr", use_video_port=USE_VIDEO_PORT)
        frame = buf[:RESOLUTION[1],:RESOLUTION[0]] # slice off padding pixels
        assert frame.base is buf # make sure slicing doesn't copy
        value = self.convolution_nn.recoginze(frame)
        print(value)
        return value