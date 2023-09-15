#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys
import threading

import cv2
import pyrealsense2 as rs
import numpy as np
import time
import datetime

from TipTrack.utility.surface_selector import SurfaceSelector
from TipTrack.utility.surface_extractor import SurfaceExtractor


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

DEVICE_ID_LEFT = '017322072607'
DEVICE_ID_RIGHT = '020522070383'

IR_RES_X = 848
IR_RES_Y = 480
IR_FPS = 90

LASER_POWER = 0  # 0 - 360

SET_ROI = False
SET_AUTO_EXPOSURE = False

IR_SENSOR_EXPOSURE = 1200  #5000  # 1500  #  1800#900 # 1800
IR_SENSOR_GAIN = 16 #32   # 200 #100  # 200


EXPOSURE_CALIBRATION_MODE = 10000  # Really bright to help see the corner markers
GAIN_CALIBARATION_MODE = 20000

NUM_FRAMES_WAIT_INITIALIZING = 30  # Let the camera warm up and let the auto white balance adjust

DEBUG_MODE = False
CALIBRATION_MODE = True
EXTRACT_PROJECTION_AREA = False


CALIBRATION_DATA_PATH = ''

surface_extractor = SurfaceExtractor()

class RealsenseD435Camera:

    num_frame = 0
    start_time = time.time()

    left_ir_image_1 = None
    left_ir_image_2 = None

    camera_matrix_ir = None
    dist_matrix_ir = None

    rectify_maps = []

    ir_sensor_exposure = IR_SENSOR_EXPOSURE
    ir_sensor_gain = IR_SENSOR_GAIN

    current_exposure = IR_SENSOR_EXPOSURE
    current_gain = IR_SENSOR_GAIN

    camera_serial_numbers = [DEVICE_ID_LEFT, DEVICE_ID_RIGHT]

    def __init__(self, subscriber=None):

        # self.load_camera_calibration_data()
        self.subscriber = subscriber

        global surface_extractor
        matrix_1 = surface_extractor.get_homography(IR_RES_X, IR_RES_Y, DEVICE_ID_LEFT)
        matrix_2 = surface_extractor.get_homography(IR_RES_X, IR_RES_Y, DEVICE_ID_RIGHT)
        self.matrices = [matrix_1, matrix_2]

        self.init_video_capture()
        self.update()

    def load_camera_calibration_data(self):
        print('load calibration data')

        for i, camera_serial_number in enumerate(self.camera_serial_numbers):
            try:
                # TODO: Change path
                cv_file = cv2.FileStorage(
                    os.path.join(CALIBRATION_DATA_PATH, 'Realsense D435 {}.yml'.format(camera_serial_number)),
                    cv2.FILE_STORAGE_READ)
                camera_matrix = cv_file.getNode('K').mat()
                dist_matrix = cv_file.getNode('D').mat()

                map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_matrix, None, None,
                                                         (IR_RES_X, IR_RES_Y), cv2.CV_32FC1)

                self.rectify_maps.append([map1, map2])

                cv_file.release()
            except Exception as e:
                print('Error in load_camera_calibration_data():', e)
                print('Cant load calibration data for Flir Blackfly S {}.yml'.format(i))

    def init_video_capture(self):
        try:
            # Create a pipeline
            self.pipeline = rs.pipeline()

            config = rs.config()
            config.enable_device(DEVICE_ID_LEFT)
            config.enable_stream(rs.stream.infrared, 1, IR_RES_X, IR_RES_Y, rs.format.y8, IR_FPS)
            # config.enable_stream(rs.stream.infrared, 2, IR_RES_X, IR_RES_Y, rs.format.y8, IR_FPS)

            # Start streaming
            profile = self.pipeline.start(config)

            print(profile.get_device())

            self.depth_ir_sensor = profile.get_device().first_depth_sensor()
            self.depth_ir_sensor.set_option(rs.option.laser_power, LASER_POWER)
            self.depth_ir_sensor.set_option(rs.option.enable_auto_exposure, SET_AUTO_EXPOSURE)

            if CALIBRATION_MODE:
                self.depth_ir_sensor.set_option(rs.option.exposure, EXPOSURE_CALIBRATION_MODE)
                self.depth_ir_sensor.set_option(rs.option.gain, GAIN_CALIBARATION_MODE)
            else:
                self.depth_ir_sensor.set_option(rs.option.exposure, IR_SENSOR_EXPOSURE)
                self.depth_ir_sensor.set_option(rs.option.gain, IR_SENSOR_GAIN)

            self.pipeline_2 = rs.pipeline()

            config_2 = rs.config()
            config_2.enable_device(DEVICE_ID_RIGHT)
            config_2.enable_stream(rs.stream.infrared, 1, IR_RES_X, IR_RES_Y, rs.format.y8, IR_FPS)

            # Start streaming
            profile_2 = self.pipeline_2.start(config_2)

            print(profile_2.get_device())

            self.depth_ir_sensor_2 = profile_2.get_device().first_depth_sensor()
            self.depth_ir_sensor_2.set_option(rs.option.laser_power, LASER_POWER)
            self.depth_ir_sensor_2.set_option(rs.option.enable_auto_exposure, SET_AUTO_EXPOSURE)

            if CALIBRATION_MODE:
                self.depth_ir_sensor_2.set_option(rs.option.exposure, EXPOSURE_CALIBRATION_MODE)
                self.depth_ir_sensor_2.set_option(rs.option.gain, GAIN_CALIBARATION_MODE)
            else:
                self.depth_ir_sensor_2.set_option(rs.option.exposure, IR_SENSOR_EXPOSURE)
                self.depth_ir_sensor_2.set_option(rs.option.gain, IR_SENSOR_GAIN)

            if DEBUG_MODE:
                intrinsics = str(profile.get_stream(rs.stream.infrared).as_video_stream_profile().get_intrinsics())
                print('[RealsenseD435Camera]: Intrinsics:', intrinsics)

        except Exception as e:
            print('[RealsenseD435Camera]: ERROR:', e, file=sys.stderr)
            print('[RealsenseD435Camera]: Could not initialize camera. If the resource is busy, check if any other '
                  'script is currently accessing the camera. If this is not the case, replug the camera and try again.',
                  file=sys.stderr)
            sys.exit(0)

    def update(self):
        print('[RealsenseD435Camera]: Skip first ' + str(NUM_FRAMES_WAIT_INITIALIZING) +
              ' frames to allow Auto White Balance to adjust')

        while True:
            self.process_frame()

    # @timeit("RealCam")
    def process_frame(self):
        self.num_frame += 1

        # TODO: Two process_frame functions
        frames = self.pipeline.wait_for_frames()
        left_ir_image_1 = frames.get_infrared_frame(1)

        frames_2 = self.pipeline_2.wait_for_frames()
        left_ir_image_2 = frames_2.get_infrared_frame(1)

        if not left_ir_image_1:
            return

        if self.num_frame < NUM_FRAMES_WAIT_INITIALIZING:
            return
        elif self.num_frame == NUM_FRAMES_WAIT_INITIALIZING:
            print('[RealsenseD435Camera]: Camera Ready')

        left_ir_image_1 = np.asanyarray(left_ir_image_1.get_data())
        left_ir_image_2 = np.asanyarray(left_ir_image_2.get_data())

        if DEBUG_MODE:
            cv2.imshow('Raw IR roi 1', left_ir_image_1)
            cv2.imshow('Raw IR roi 2', left_ir_image_2)

        # Undistort camera images
        if len(self.rectify_maps) > 0:
            left_ir_image_1 = cv2.remap(left_ir_image_1, self.rectify_maps[0][0], self.rectify_maps[0][1], interpolation=cv2.INTER_LINEAR)
            left_ir_image_2 = cv2.remap(left_ir_image_2, self.rectify_maps[1][0], self.rectify_maps[1][1], interpolation=cv2.INTER_LINEAR)

            # if DEBUG_MODE:
            #     cv2.imshow('ir after undistort', left_ir_image_1)

        if DEBUG_MODE:
            # img_preview = cv2.cvtColor(ir_image_table, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('test', img_preview)
            cv2.waitKey(1)

        self.subscriber.on_new_frame_group([left_ir_image_1, left_ir_image_2], self.camera_serial_numbers, self.matrices)


class CameraTester:

    frames = []
    camera_serial_numbers = []

    windows_initialized = False

    def __init__(self):
        global DEBUG_MODE
        global EXTRACT_PROJECTION_AREA

        # If this script is started as main, the debug mode is activated by default:
        DEBUG_MODE = True
        EXTRACT_PROJECTION_AREA = False


        # from ir_pen import IRPen
        # self.ir_pen = IRPen()

        thread = threading.Thread(target=self.debug_mode_thread)
        thread.start()
        # thread.join()

        if CALIBRATION_MODE:
            DEBUG_MODE = False  # No Debug Mode wanted in Calibration mode
            self.surface_selector = SurfaceSelector()

        realsense_d435 = RealsenseD435Camera(subscriber=self)

        # TODO: Improve
        time.sleep(100000)  # Wait before it stops
        print('ENDING CAMERA TESTING')

    def debug_mode_thread(self):
        while True:

            extracted_frames = []

            if not self.windows_initialized:
                if len(self.camera_serial_numbers) > 0:
                    for camera_serial_number in self.camera_serial_numbers:
                        cv2.namedWindow('Realsense D435 {}'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Realsense D435 {}'.format(camera_serial_number), IR_RES_X, IR_RES_Y)
                        # cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), 480, 480)

                        if EXTRACT_PROJECTION_AREA:
                            cv2.namedWindow('Realsense D435 {} Extracted'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Realsense D435 {} Extracted'.format(camera_serial_number), IR_RES_X, IR_RES_Y)
                            # cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), 480, 480)

                            # cv2.namedWindow('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN)
                            # cv2.setWindowProperty('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN,
                            #                       cv2.WINDOW_FULLSCREEN)
                    self.windows_initialized = True
                else:
                    continue

            if len(self.frames) > 0:
                for i, frame in enumerate(self.frames):

                    window_name = 'Realsense D435 {}'.format(self.camera_serial_numbers[i])
                    window_name_extracted = 'Realsense D435 {} Extracted'.format(self.camera_serial_numbers[i])

                    # pen_event_roi, brightest, (x, y) = self.ir_pen.crop_image(frame)

                    if DEBUG_MODE:
                        cv2.imshow(window_name, frame)
                        # cv2.imshow(window_name, pen_event_roi)

                    if EXTRACT_PROJECTION_AREA:
                        global surface_extractor
                        extracted_frame = surface_extractor.extract_table_area(frame, window_name)
                        extracted_frame = cv2.resize(extracted_frame, (3840, 2160))
                        extracted_frames.append(extracted_frame)
                        cv2.imshow(window_name_extracted, extracted_frame)

                    if CALIBRATION_MODE:
                            calibration_finished = self.surface_selector.select_surface(frame, window_name)

                            if calibration_finished:
                                print('[Surface Selector Node]: Calibration finished for {}'.format(window_name))

                self.frames = []

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                sys.exit(0)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(self.camera_serial_numbers) == 0:
            self.camera_serial_numbers = camera_serial_numbers

        self.frames = frames


if __name__ == '__main__':
    CameraTester()
