#!/usr/bin/env
# coding: utf-8
import sys

# Code parts for asynchronous video capture based on:
# http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/

import cv2
import threading
import time

from TipTrack.utility.surface_selector import SurfaceSelector
from TipTrack.utility.surface_extractor import SurfaceExtractor

# TODO: Automatically set exposure time and other parameter (currently we do this manually in GUVCVIEW)

CAMERA_ID = 0
RES_X = 1280  # 1920  #3840  # 1280#3840#4096#
RES_Y = 720  # 1080  # 2160  # 720#2160#2160#
FPS = 60  # 60  # 30  # 15

CALIBRATION_MODE = False
CAMERA_PARAMETER_NAME = 'Logitech Brio'


class LogitechBrio:

    frame = None
    counter = 0
    start_time = time.time()
    thread = None

    homography_matrix = None

    def __init__(self, subscriber):
        self.started = False
        surface_extractor = SurfaceExtractor()
        self.homography_matrix = surface_extractor.get_homography(RES_X, RES_Y, CAMERA_PARAMETER_NAME)

        self.subscriber = subscriber

    def init_video_capture(self, camera_id=CAMERA_ID, resolution_x=RES_X, resolution_y=RES_Y, fps=FPS):
        self.capture = cv2.VideoCapture()
        self.capture.open(camera_id)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def start(self):
        if self.started:  # Prevent the thread from starting it again if it is already running
            print('Already running')
            return None
        else:
            self.started = True
            self.thread = threading.Thread(target=self.__update, args=())
            # thread.daemon = True
            self.thread.start()
            return self

    def __update(self):
        while self.started:
            # Get the newest frame from the camera
            ret, frame = self.capture.read()

            # If a new frame is available, store it in the corresponding variable
            if frame is not None:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.counter += 1
                if frame.shape[0] < RES_Y or frame.shape[1] < RES_X:
                    print('output image shape: ', frame.shape)
                    print('WARNING: Output image resolution for additional camera is smaller then expected!')
                # with self.read_lock:
                if self.subscriber is not None:
                    self.subscriber.on_new_color_frame(frame, self.homography_matrix)
                # self.frame = frame
                if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
                    # print("FPS: %s" % round(self.counter / (time.time() - self.start_time), 1))
                    self.counter = 0
                    self.start_time = time.time()

    # Stop the thread
    def stop(self):
        if self.started:
            self.started = False
            self.thread.join()

    # Release the camera if the script is stopped
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()
        

class LogitechBrioDebugger:
    
    def __init__(self):
        self.surface_selector = SurfaceSelector()

        self.surface_extractor = SurfaceExtractor()

        camera = LogitechBrio(self)
        camera.init_video_capture()
        camera.start()
        self.pen_detector = PenColorDetector()

        self.main_loop()

    def main_loop(self):
        while True:
            time.sleep(1)

    def on_new_color_frame(self, frame, homography_matrix):
        if CALIBRATION_MODE:
            calibration_finished = self.surface_selector.select_surface(frame, CAMERA_PARAMETER_NAME)

            if calibration_finished:
                print('[Surface Selector Node]: Calibration finished for {}'.format(CAMERA_PARAMETER_NAME))
                sys.exit(0)
            print(frame.shape)
        elif frame is not None:
            cv2.imshow('Logitech Brio', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    debugger = LogitechBrioDebugger()
