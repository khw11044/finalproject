import cv2
from threading import Thread
import time
import numpy as np

class VideoStream:
    def __init__(self,resolution=(640,640),framerate=30):
        print("init")
        self.stream = cv2.VideoCapture(0)
        #self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        #self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        time.sleep(2.0)
    
    def start(self):
        print("start thread")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        print("read")
        while True:
            if self.stopped:
                return
            
            (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True


