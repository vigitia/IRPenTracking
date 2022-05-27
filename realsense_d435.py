#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code parts for asynchronous video capture taken from
# http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/

# Code parts for the RealSense Camera taken from
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
import os.path
import sys

import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import time

import datetime

from laser_pen_detector_service import LaserPenDetectorService


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            print("I " +prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time =  datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) +" ms)")
            return retval
        return wrapper
    return timeit_decorator




# Camera Settings
DEPTH_RES_X = 1280
DEPTH_RES_Y = 720
DEPTH_FPS = 30

RGB_RES_X = 848
RGB_RES_Y = 480
RGB_FPS = 90

IR_RES_X = 848
IR_RES_Y = 480
IR_FPS = 90

COLORIZER_MIN_DISTANCE = 0.5  # m
COLORIZER_MAX_DISTANCE = 1.5  # m

LASER_POWER = 0  # 0 - 360
SET_ROI = False
ROI_MIN = (305, 235)
ROI_MAX = (338, 254)

IR_SENSOR_EXPOSURE = 1800  # 1500  #  1800#900 # 1800
IR_SENSOR_GAIN = 200   # 200 #100  # 200

RGB_SENSOR_EXPOSURE = 400
RGB_SENSOR_GAIN = 20

NUM_FRAMES_WAIT_INITIALIZING = 30  # Let the camera warm up and let the auto white balance adjust

ENABLE_RGB = False

DEBUG_MODE = False
# TODO: Add Debug mode

# # For undistorting the RGB camera
# mtx = np.array([[1.40149086e+03, 0.00000000e+00, 9.99353256e+02],
#                 [0.00000000e+00, 1.39935340e+03, 5.48174317e+02],
#                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
# dist = np.array([[2.03046975e-01, -6.68070627e-01, 4.20827516e-03, 5.45944400e-04, 6.58760428e-01]])

CALIBRATION_DATA_PATH = ''


class RealsenseD435Camera:

    num_frame = 0
    frame_counter = 0
    start_time = time.time()

    pipeline = None
    align_to_color = None
    colorizer = None

    color_image = None
    depth_image = None
    left_ir_image = None
    new_frames = False

    camera_matrix_rgb = None
    dist_matrix_rgb = None

    camera_matrix_ir = None
    dist_matrix_ir = None

    def __init__(self):

        self.load_camera_calibration_data()

        self.laser_pen_detector = LaserPenDetectorService()

        # print(self.camera_matrix_rgb, self.camera_matrix_ir)

    def load_camera_calibration_data(self):
        print('load calibration data')
        try:
            cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, '{}.yml'.format('rgb_full')), cv2.FILE_STORAGE_READ)
            self.camera_matrix_rgb = cv_file.getNode('K').mat()
            self.dist_matrix_rgb = cv_file.getNode('D').mat()
            cv_file.release()
        except:
            print('Cant load calibration data for rgb sensor')

        try:
            cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, '{}.yml'.format('ir_full')), cv2.FILE_STORAGE_READ)
            self.camera_matrix_ir = cv_file.getNode('K').mat()
            self.dist_matrix_ir = cv_file.getNode('D').mat()
            cv_file.release()
        except:
            print('Cant load calibration data for ir sensor')

    def init_video_capture(self):
        try:
            # Create a pipeline
            self.pipeline = rs.pipeline()

            # Create a config and configure the pipeline to stream different resolutions of color and depth streams
            config = rs.config()
            if ENABLE_RGB:
                config.enable_stream(rs.stream.color, RGB_RES_X, RGB_RES_Y, rs.format.bgr8, RGB_FPS)
            # config.enable_stream(rs.stream.depth, DEPTH_RES_X, DEPTH_RES_Y, rs.format.z16, DEPTH_FPS)
            config.enable_stream(rs.stream.infrared, 1, IR_RES_X, IR_RES_Y, rs.format.y8, IR_FPS)
            # config.enable_stream(rs.stream.infrared, 2, DEPTH_RES_X, DEPTH_RES_Y, rs.format.y8, DEPTH_FPS)

            # Start streaming
            profile = self.pipeline.start(config)

            if ENABLE_RGB:
                rgb_sensor = profile.get_device().first_color_sensor()
                rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
                rgb_sensor.set_option(rs.option.exposure, RGB_SENSOR_EXPOSURE)
                rgb_sensor.set_option(rs.option.gain, RGB_SENSOR_GAIN)

            # sensor = profile.get_device().query_sensors()[0]
            depth_ir_sensor = profile.get_device().first_depth_sensor()
            depth_ir_sensor.set_option(rs.option.laser_power, LASER_POWER)  # 0 - 360

            depth_ir_sensor.set_option(rs.option.enable_auto_exposure, 0)

            depth_ir_sensor.set_option(rs.option.exposure, IR_SENSOR_EXPOSURE)
            depth_ir_sensor.set_option(rs.option.gain, IR_SENSOR_GAIN)
            print(depth_ir_sensor.get_option(rs.option.exposure), depth_ir_sensor.get_option(rs.option.gain))


            # depth_ir_sensor.set_option(rs.option.exposure, 1)

            # Setup ROI
            # if SET_ROI:
            #     s = profile.get_device().first_roi_sensor()
            #     roi = rs.region_of_interest()
            #     roi.min_x = ROI_MIN[0]
            #     roi.min_y = ROI_MIN[1]
            #     roi.max_x = ROI_MAX[0]
            #     roi.max_y = ROI_MAX[1]
            #     print('[RealsenseD435Camera] ROI of Realsense camera: ({}, {}) ({}, {})'.format(roi.min_x, roi.min_y, roi.max_x, roi.max_y))
            #     s.set_region_of_interest(roi)

            # # Getting the depth sensor's depth scale (see rs-align example for explanation)
            # self.depth_scale = depth_sensor.get_depth_scale()
            # if DEBUG_MODE:
            #     print('[RealsenseD435Camera]: Depth scale:', self.depth_scale)
            #
            # # Number of meters represented by a single depth unit. DO NOT CHANGE!
            # depth_sensor.set_option(rs.option.depth_units, 0.001)

            # rs.align allows us here to perform alignment of depth frames to color frames
            # The param is the stream type to which we plan to align depth frames.
            # self.align_to_color = rs.align(rs.stream.color)
            # self.align_to_depth = rs.align(rs.stream.depth)

            # Init filters for depth image
            # self.hole_filling_filter = rs.hole_filling_filter()
            # self.decimation_filter = rs.decimation_filter()
            # self.temporal_filter = rs.temporal_filter()

            if DEBUG_MODE:
                intrinsics = str(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
                print('[RealsenseD435Camera]: Intrinsics:', intrinsics)

        except Exception as e:
            print('[RealsenseD435Camera]: ERROR:', e, file=sys.stderr)
            print('[RealsenseD435Camera]: Could not initialize camera. If the resource is busy, check if any other '
                  'script is currently accessing the camera. If this is not the case, replug the camera and try again.',
                  file=sys.stderr)
            sys.exit(0)

        # self.init_colorizer()

        self.started = False
        self.read_lock = threading.Lock()

    # The colorizer can colorize depth images
    def __init_colorizer(self):
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)  # Define the color scheme
        # Auto histogram color selection (0 = off, 1 = on)
        self.colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
        self.colorizer.set_option(rs.option.min_distance, COLORIZER_MIN_DISTANCE)  # meter
        self.colorizer.set_option(rs.option.max_distance, COLORIZER_MAX_DISTANCE)  # meter

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

    @timeit("RealCam")
    def process_frame(self):
        self.num_frame += 1
        if DEBUG_MODE:
            print('[RealsenseD435Camera]: Frame ', self.num_frame)

        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        # aligned_frames_to_color = self.align_to_color.process(frames)

        # Get aligned frames
        # color_frame = aligned_frames_to_color.get_color_frame()
        # aligned_depth_frame = aligned_frames_to_color.get_depth_frame()

        color_frame = None
        left_ir_image = None

        if ENABLE_RGB:
            color_frame = frames.get_color_frame()
        left_ir_image = frames.get_infrared_frame(1)
        # right_ir_image = frames.get_infrared_frame(2)

        # Validate that both frames are valid
        if ENABLE_RGB and not color_frame:
            return

        if not left_ir_image:  # or not aligned_depth_frame:
            return

        if self.num_frame < NUM_FRAMES_WAIT_INITIALIZING:
            return
        elif self.num_frame == NUM_FRAMES_WAIT_INITIALIZING:
            print('[RealsenseD435Camera]: Camera Ready')

        # Apply Filters
        # aligned_depth_frame = self.__apply_filters(aligned_depth_frame)

        # Convert frames
        if ENABLE_RGB:
            color_image = np.asanyarray(color_frame.get_data())
        # color_image = None
        # depth_image = np.array(aligned_depth_frame.get_data(), dtype=np.uint16)
        # depth_image = self.__get_depth_image_mm(depth_image)
        depth_image = None
        # depth_colormap = np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data())
        left_ir_image = np.asanyarray(left_ir_image.get_data())
        # right_ir_image = np.asanyarray(ir_right_frame.get_data())

        # Undistort camera images
        if ENABLE_RGB and self.camera_matrix_rgb is not None and self.dist_matrix_rgb is not None:
            color_image = cv2.undistort(color_image, self.camera_matrix_rgb, self.dist_matrix_rgb, None, None)
        if self.camera_matrix_ir is not None and self.dist_matrix_ir is not None:
            left_ir_image = cv2.undistort(left_ir_image, self.camera_matrix_ir, self.dist_matrix_ir, None, None)

        # Adjust IR frames to the same size as the RGB image
        # TODO: This would not be needed if the TableExtractionService could deal with different frame sizes
        # if color_image.shape[0] != left_ir_image.shape[0]:
        #     left_ir_image = cv2.resize(left_ir_image, (color_image.shape[1], color_image.shape[0]),
        #                                interpolation=cv2.INTER_AREA)
        self.frame_counter += 1
        with self.read_lock:
            if ENABLE_RGB:
                self.color_image = color_image
            self.depth_image = depth_image
            self.left_ir_image = left_ir_image
            # self.right_ir_image = right_ir_image
            # self.depth_colormap = depth_colormap

            self.new_frames = True

            self.color_image = None
            self.change_color = False


            pen_spots, stored_lines, new_lines, points_to_remove, selected_color = self.laser_pen_detector.get_pen_spots(
                self.color_image, self.left_ir_image, self.change_color)

            if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
                print("FPS: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
                self.frame_counter = 0
                self.start_time = time.time()

    def __apply_filters(self, aligned_depth_frame):
        aligned_depth_frame = self.hole_filling_filter.process(aligned_depth_frame)
        aligned_depth_frame = self.decimation_filter.process(aligned_depth_frame)
        aligned_depth_frame = self.temporal_filter.process(aligned_depth_frame)

        return aligned_depth_frame

    # Convert the depth image into a numpy array where each pixel value corresponds to the measured distance in mm
    # This conversion only works if the depth units are set to 0.001
    def __get_depth_image_mm(self, depth_image):
        depth_image_mm = depth_image.copy() * self.depth_scale * 1000
        depth_image_mm = np.array(depth_image_mm, dtype=np.uint16)

        return depth_image_mm

    # Returns the requested camera frames
    # TODO: Return only the frames that are requested via params
    def get_frames(self):
        with self.read_lock:
            if self.new_frames:
                if self.color_image is not None:  # and self.depth_image is not None:
                    self.new_frames = False
                    return self.color_image, self.depth_image, self.left_ir_image
            return None, None, None

    # Just a test
    def confirm_received_frames(self):
        with self.read_lock:
            # print('confirm received frame')
            self.new_frames = False
            self.color_image = None
            self.depth_image = None
            self.left_ir_image = None

    def get_resolution(self):
        return RGB_RES_X, RGB_RES_Y

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
