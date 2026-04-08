# S1_main 내장용 - Pi CSI 카메라 (camera_id=70)
import time
import threading
import os
import cv2

_BLOCK_DIR = os.path.dirname(os.path.abspath(__file__))
camera_handle = None

class Camera(object):
    thread = None
    frame = None
    last_access = 0

    def __init__(self, camera_id=70):
        global camera_handle
        self._camera_id = camera_id
        if camera_handle is None:
            c = cv2.VideoCapture(camera_id)
            c.set(cv2.CAP_PROP_FPS, 15)
            camera_handle = c

    def initialize(self):
        global camera_handle
        if Camera.thread is None:
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()
            while self.frame is None:
                time.sleep(0.01)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return cv2.flip(Camera.frame, -1) if Camera.frame is not None else None

    @classmethod
    def _thread(cls):
        global camera_handle
        while True:
            ret, cls.frame = camera_handle.read()
            if not ret:
                break
        cls.thread = None
