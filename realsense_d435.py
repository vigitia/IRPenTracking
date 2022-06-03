#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code parts for asynchronous video capture taken from
# http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/

import os.path
import sys
import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import time
import random
import datetime

from surface_selector import SurfaceSelector
from table_extraction_service import TableExtractionService


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            # print("I " + prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            # print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) + " ms)")
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return retval
        return wrapper
    return timeit_decorator


# Camera Settings
IR_RES_X = 848
IR_RES_Y = 480
IR_FPS = 90

LASER_POWER = 0  # 0 - 360

SET_ROI = False
SET_AUTO_EXPOSURE = False

IR_SENSOR_EXPOSURE = 3000  # 1500  #  1800#900 # 1800
IR_SENSOR_GAIN = 300   # 200 #100  # 200

IR_SENSOR_EXPOSURE_MAX = 4000
IR_SENSOR_EXPOSURE_MIN = 250
IR_SENSOR_GAIN_MAX = 400
IR_SENSOR_GAIN_MIN = 25

NUM_FRAMES_WAIT_INITIALIZING = 30  # Let the camera warm up and let the auto white balance adjust

DEBUG_MODE = False
# TODO: Add Debug mode


CALIBRATION_DATA_PATH = ''
CALIBRATION_MODE = False
EXPOSURE_CALIBRATION_MODE = False
EXPOSURE_CALIBRATION_MODE_2 = True
CAMERA_PATH = '/vigitia/realsense_ir_full'
depth_ir_sensor = None
write_counter = 0
RECORD_MODE = False
TRAINING_CONDITION = 'hover_sunlight'
TRAINING_PATH = 'hover_sunlight'
PREDICTION_MODE = True

img_drawing = None


class RealsenseD435Camera:

    num_frame = 0
    start_time = time.time()

    pipeline = None

    ir_image_cropped = None
    new_frames = False

    camera_matrix_ir = None
    dist_matrix_ir = None

    img_id = 0

    ir_sensor_exposure = IR_SENSOR_EXPOSURE
    ir_sensor_gain = IR_SENSOR_GAIN
    exposure_calibration_mode = EXPOSURE_CALIBRATION_MODE
    exposure_calibration_mode_2 = EXPOSURE_CALIBRATION_MODE_2

    def __init__(self):
        self.load_camera_calibration_data()

        self.surface_selector = SurfaceSelector(CAMERA_PATH)
        self.table_extractor = TableExtractionService()

    def load_camera_calibration_data(self):
        print('load calibration data')

        try:
            # TODO: Change path
            cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, '{}.yml'.format('ir_full')), cv2.FILE_STORAGE_READ)
            self.camera_matrix_ir = cv_file.getNode('K').mat()
            self.dist_matrix_ir = cv_file.getNode('D').mat()
            cv_file.release()
        except:
            print('Cant load calibration data for ir sensor')

    def init_video_capture(self):
        # global depth_ir_sensor
        try:
            # Create a pipeline
            self.pipeline = rs.pipeline()

            config = rs.config()
            config.enable_stream(rs.stream.infrared, 1, IR_RES_X, IR_RES_Y, rs.format.y8, IR_FPS)

            # Start streaming
            profile = self.pipeline.start(config)

            self.depth_ir_sensor = profile.get_device().first_depth_sensor()
            self.depth_ir_sensor.set_option(rs.option.laser_power, LASER_POWER)
            self.depth_ir_sensor.set_option(rs.option.enable_auto_exposure, SET_AUTO_EXPOSURE)
            self.depth_ir_sensor.set_option(rs.option.exposure, IR_SENSOR_EXPOSURE)
            self.depth_ir_sensor.set_option(rs.option.gain, IR_SENSOR_GAIN)
            # print(depth_ir_sensor.get_option(rs.option.exposure), depth_ir_sensor.get_option(rs.option.gain))

            if DEBUG_MODE:
                intrinsics = str(profile.get_stream(rs.stream.infrared).as_video_stream_profile().get_intrinsics())
                print('[RealsenseD435Camera]: Intrinsics:', intrinsics)

        except Exception as e:
            print('[RealsenseD435Camera]: ERROR:', e, file=sys.stderr)
            print('[RealsenseD435Camera]: Could not initialize camera. If the resource is busy, check if any other '
                  'script is currently accessing the camera. If this is not the case, replug the camera and try again.',
                  file=sys.stderr)
            sys.exit(0)

        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        else:
            self.started = True
            self.thread = threading.Thread(target=self.update, args=())
            # thread.daemon = True
            self.thread.start()
            return self

    def update(self):
        print('[RealsenseD435Camera]: Skip first ' + str(NUM_FRAMES_WAIT_INITIALIZING) +
              ' frames to allow Auto White Balance to adjust')

        while self.started:
            self.process_frame()

    # @timeit("RealCam")
    def process_frame(self):
        self.num_frame += 1

        frames = self.pipeline.wait_for_frames()

        left_ir_image = frames.get_infrared_frame(1)

        if not left_ir_image:  # or not aligned_depth_frame:
            return

        if self.num_frame < NUM_FRAMES_WAIT_INITIALIZING:
            return
        elif self.num_frame == NUM_FRAMES_WAIT_INITIALIZING:
            print('[RealsenseD435Camera]: Camera Ready')

        left_ir_image = np.asanyarray(left_ir_image.get_data())

        # Undistort camera images
        if self.camera_matrix_ir is not None and self.dist_matrix_ir is not None:
            left_ir_image = cv2.undistort(left_ir_image, self.camera_matrix_ir, self.dist_matrix_ir, None, None)

        if CALIBRATION_MODE:
            #print(self.left_ir_image.shape)
            calibration_finished = self.surface_selector.select_surface(left_ir_image)
            calibration_finished = False

            cv2.waitKey(1)

            if calibration_finished:
                print("[Surface Selector Node]: Calibration Finished")
                exit()
        else:

            ir_image_table = self.table_extractor.extract_table_area(left_ir_image, CAMERA_PATH)
            with self.read_lock:
                self.ir_image_cropped = ir_image_table
                self.new_frames = True

            if self.exposure_calibration_mode:
                max_brightness = np.max(ir_image_table)
                #print(self.ir_sensor_exposure, max_brightness)
                if max_brightness > 240:
                    if self.ir_sensor_exposure > 50:
                        self.ir_sensor_exposure -= 50
                        self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
                    if self.ir_sensor_gain > 50:
                        self.ir_sensor_gain -= 50
                        self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
                    print(f'gain: {self.ir_sensor_gain}, exposure: {self.ir_sensor_exposure}')
                elif max_brightness < 100:
                    self.ir_sensor_exposure += 50
                    self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
                else:
                    print(f'exposure: {self.ir_sensor_exposure}, gain: {self.ir_sensor_gain}, max: {np.max(ir_image_table)}')
                    self.exposure_calibration_mode = False
            elif self.exposure_calibration_mode_2:
                mean_brightness = np.mean(ir_image_table)
                #print(self.ir_sensor_exposure, max_brightness)
                if mean_brightness > 45:
                    if self.ir_sensor_exposure > 50:
                        self.ir_sensor_exposure -= 50
                        self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
                    if self.ir_sensor_gain > 35:
                        self.ir_sensor_gain -= 50
                        self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
                    print(f'gain: {self.ir_sensor_gain}, exposure: {self.ir_sensor_exposure}')
                elif mean_brightness < 25:
                    self.ir_sensor_exposure += 50
                    self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
                else:
                    print(f'exposure: {self.ir_sensor_exposure}, gain: {self.ir_sensor_gain}, max: {np.max(ir_image_table)}')
                    self.exposure_calibration_mode_2 = False

            if RECORD_MODE:
                # global img_id, write_counter
                #write_coutner = 0
                cv2.imwrite(f'out/{TRAINING_PATH}/{self.img_id:04d}_{TRAINING_CONDITION}_{self.ir_sensor_exposure}_{self.ir_sensor_gain}.png', ir_image_table)
                self.ir_sensor_exposure = random.randint(IR_SENSOR_EXPOSURE_MIN, IR_SENSOR_EXPOSURE_MAX)
                self.ir_sensor_gain = random.randint(IR_SENSOR_GAIN_MIN, IR_SENSOR_GAIN_MAX)
                self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
                self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
                time.sleep(0.05)

                self.img_id += 1
                #time.sleep(1)

            img_preview = cv2.cvtColor(ir_image_table, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('test', img_preview)
            # cv2.waitKey(1)

    # Returns the requested camera frames
    def get_ir_image(self):
        with self.read_lock:
            if self.new_frames and self.ir_image_cropped is not None:
                self.new_frames = False
                return self.ir_image_cropped
            return None

    def stop(self):
        self.started = False
        self.thread.join()
        self.pipeline.stop()

    def __exit__(self, exec_type, exc_value, traceback):
        self.pipeline.stop()


if __name__ == '__main__':
    realsense_d435_camera = RealsenseD435Camera()
    realsense_d435_camera.init_video_capture()
    realsense_d435_camera.start()



