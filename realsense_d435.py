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

from scipy.spatial import distance

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

IR_SENSOR_EXPOSURE = 1200  #5000  # 1500  #  1800#900 # 1800
IR_SENSOR_GAIN = 16 #32   # 200 #100  # 200

IR_SENSOR_EXPOSURE_RIGHT = 1200
IR_SENSOR_GAIN_RIGHT = 16

# IR_SENSOR_EXPOSURE_MAX = 4000
# IR_SENSOR_EXPOSURE_MIN = 250
# IR_SENSOR_GAIN_MAX = 400
# IR_SENSOR_GAIN_MIN = 25

EXPOSURE_CALIBRATION_MODE = 10000  # Really bright to help see the corner markers
GAIN_CALIBARATION_MODE = 20000

NUM_FRAMES_WAIT_INITIALIZING = 30  # Let the camera warm up and let the auto white balance adjust

# Cut out the calibrated area from the frame
EXTRACT_PROJECTION_AREA = True

DEBUG_MODE = False


CALIBRATION_DATA_PATH = ''
CALIBRATION_MODE = False
TRAINING_DATA_COLLECTION_MODE = False
# AUTO_EXPOSURE_1 = False
# AUTO_EXPOSURE_2 = False
CAMERA_PATH_0 = '/vigitia/realsense_ir_full_0'
CAMERA_PATH_1 = '/vigitia/realsense_ir_full_1'
CAMERA_PATH_2 = '/vigitia/realsense_ir_full_2'
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

    current_ir_image = None
    current_ir_image_table_0 = None
    current_ir_image_table_1 = None
    current_ir_image_table_2 = None
    left_ir_image_0 = None
    left_ir_image_1 = None
    left_ir_image_2 = None

    matrix_1 = None
    matrix_2 = None

    new_frames = False

    camera_matrix_ir = None
    dist_matrix_ir = None

    img_id = 0

    ir_sensor_exposure = IR_SENSOR_EXPOSURE
    ir_sensor_gain = IR_SENSOR_GAIN
    # exposure_calibration_mode = AUTO_EXPOSURE_1
    # exposure_calibration_mode_2 = AUTO_EXPOSURE_2

    # permutations = [(100, 16), (200, 16), (400, 16), (800, 16), (1600, 16), (3200, 16), (6400, 16), (10000, 16),
    #                 (10000, 32), (10000, 64), (10000, 128), (10000, 248)]

    # permutations = [(10000, 16), (10000, 32), (10000, 64)]
    # permutations = [(5000, 64), (2500, 32)]
    permutations = [(800, 16), (1200, 16)]

    permutation_index = 0
    current_exposure = IR_SENSOR_EXPOSURE
    current_gain = IR_SENSOR_GAIN

    def __init__(self, extract_projection_area=EXTRACT_PROJECTION_AREA):

        self.extract_projection_area = extract_projection_area

        self.load_camera_calibration_data()

        self.surface_selector = SurfaceSelector()
        self.table_extractor = TableExtractionService()

        if TRAINING_DATA_COLLECTION_MODE:
            time.sleep(5)

        # self.bias_image = cv2.imread('bias_frame.png', cv2.IMREAD_GRAYSCALE)
        # max = np.max(self.bias_image)
        # print(max)
        # self.bias_image = cv2.bitwise_not(self.bias_image)
        # self.bias_image -= (255-max)

    def load_camera_calibration_data(self):
        print('load calibration data')

        try:
            # TODO: Change path
            cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, '{}.yml'.format('ir_full')), cv2.FILE_STORAGE_READ)
            self.camera_matrix_ir = cv_file.getNode('K').mat()
            self.dist_matrix_ir = cv_file.getNode('D').mat()
            cv_file.release()
            print(self.camera_matrix_ir)
        except:
            print('Cant load calibration data for ir sensor')

    # # in case the brightness distribution of the roi is uneven,
    # # it is recommended to record an empty bias frame and use it
    # # to normalize the brightness distribution over the roi
    # def overlay_bias_image(self, original_image, alpha=0.5):
    #     result = cv2.addWeighted(original_image, alpha, self.bias_image, 1.0 - alpha, 0.0)
    #     return result

    def init_video_capture(self):
        # global depth_ir_sensor
        try:
            # Create a pipeline
            self.pipeline = rs.pipeline()

            config = rs.config()
            config.enable_device('017322072607')
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
                self.depth_ir_sensor.set_option(rs.option.exposure, IR_SENSOR_EXPOSURE_RIGHT)
                self.depth_ir_sensor.set_option(rs.option.gain, IR_SENSOR_GAIN_RIGHT)

            self.pipeline_2 = rs.pipeline()

            config_2 = rs.config()
            config_2.enable_device('020522070383')
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

    def get_next_camera_setting_values(self):

        if self.permutation_index == len(self.permutations):
            return True, (0, 0)
        next_perm = self.permutations[self.permutation_index]
        self.permutation_index += 1
        return False, next_perm

    saved_image_counter = 0

    # @timeit("RealCam")
    def process_frame(self):
        self.num_frame += 1

        # TODO: Two process_frame functions
        frames = self.pipeline.wait_for_frames()
        left_ir_image_1 = frames.get_infrared_frame(1)
        # left_ir_image_0 = frames.get_infrared_frame(2)

        frames_2 = self.pipeline_2.wait_for_frames()
        left_ir_image_2 = frames_2.get_infrared_frame(1)

        if not left_ir_image_1:
            return

        if self.num_frame < NUM_FRAMES_WAIT_INITIALIZING:
            return
        elif self.num_frame == NUM_FRAMES_WAIT_INITIALIZING:
            print('[RealsenseD435Camera]: Camera Ready')

        # left_ir_image_0 = np.asanyarray(left_ir_image_0.get_data())
        left_ir_image_1 = np.asanyarray(left_ir_image_1.get_data())
        left_ir_image_2 = np.asanyarray(left_ir_image_2.get_data())

        if DEBUG_MODE:
            # cv2.imshow('Raw IR roi 0', left_ir_image_0)
            cv2.imshow('Raw IR roi 1', left_ir_image_1)
            cv2.imshow('Raw IR roi 2', left_ir_image_2)

        # Undistort camera images
        # TODO: Get this to work for both cameras
        if self.camera_matrix_ir is not None and self.dist_matrix_ir is not None:
            left_ir_image_1 = cv2.undistort(left_ir_image_1, self.camera_matrix_ir, self.dist_matrix_ir, None, None)

            # if DEBUG_MODE:
            #     cv2.imshow('ir after undistort', left_ir_image_1)

        if self.matrix_1 is None:
            self.matrix_1 = self.table_extractor.get_homography(left_ir_image_1, CAMERA_PATH_1)

        if self.matrix_2 is None:
            self.matrix_2 = self.table_extractor.get_homography(left_ir_image_2, CAMERA_PATH_2)

        if not CALIBRATION_MODE and self.extract_projection_area:
            # ir_image_table_0, crop_0 = self.table_extractor.extract_table_area_wide(left_ir_image_0, CAMERA_PATH_0)
            #ir_image_table_1, crop_1 = self.table_extractor.extract_table_area_wide(left_ir_image_1, CAMERA_PATH_1)
            #ir_image_table_2, crop_2 = self.table_extractor.extract_table_area_wide(left_ir_image_2, CAMERA_PATH_2)

            #ir_image_table_wide_1 = self.table_extractor.extract_table_area_wide(left_ir_image_1, CAMERA_PATH_1)
            #ir_image_table_wide_2 = self.table_extractor.extract_table_area_wide(left_ir_image_2, CAMERA_PATH_2)

            # _, thresh_1 = cv2.threshold(ir_image_table_1, 80, 255, cv2.THRESH_BINARY)
            # _, thresh_2 = cv2.threshold(ir_image_table_2, 80, 255, cv2.THRESH_BINARY)
            #
            # dest_and = cv2.bitwise_and(thresh_1, thresh_2, mask=None)
            #
            # cv2.imshow('AND', dest_and)

            # zeroes = np.zeros(ir_image_table_1.shape, 'uint8')
            # fake_color = np.dstack((ir_image_table_1, ir_image_table_2, zeroes))

            # TEST!
            #fake_color_grey = cv2.cvtColor(fake_color, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('fake gray', fake_color_grey)

            # _, brightest_1, _, (max_x_1, max_y_1) = cv2.minMaxLoc(ir_image_table_1)
            # _, brightest_2, _, (max_x_2, max_y_2) = cv2.minMaxLoc(ir_image_table_2)

            # if brightest_1 > 100 or brightest_2 > 100:
            #
            #     pos1 = self.find_pen_position_subpixel(ir_image_table_1)
            #     pos2 = self.find_pen_position_subpixel(ir_image_table_2)
            #
            #     dist = distance.euclidean(pos1, pos2)
            #
            #     print(brightest_1, brightest_2, dist)
            #
            #     if dist > 40:
            #         print('No Touch!')

            # _ = self.find_pen_position_subpixel(ir_image_table_1, '1')
            # _ = self.find_pen_position_subpixel(ir_image_table_2, '2')
            # _, _ = self.find_pen_position(ir_image_table_1, '1')
            # _, _ = self.find_pen_position(ir_image_table_2, '2')

            with self.read_lock:
                # self.current_ir_image = ir_image_table_1
                #self.current_ir_image = fake_color
                # self.current_ir_image_table_0 = ir_image_table_0
                #self.current_ir_image_table_1 = ir_image_table_1
                #self.current_ir_image_table_2 = ir_image_table_2
                #self.current_ir_image_table_wide_1 = ir_image_table_wide_1
                #self.current_ir_image_table_wide_2 = ir_image_table_wide_2
                # self.left_ir_image_0 = left_ir_image_0
                self.left_ir_image_1 = left_ir_image_1
                self.left_ir_image_2 = left_ir_image_2
                #self.crop_1 = crop_1
                #self.crop_2 = crop_2
                self.new_frames = True
        # else:
        #     with self.read_lock:
        #         self.current_ir_image = left_ir_image_1
        #         self.new_frames = True

        # print(self.num_frame)
        # if self.num_frame == 200:
        #     print('Saving bias frame')
        #     cv2.imwrite('bias_frame.png', left_ir_image)

        # if DEBUG_MODE:
        #     cv2.imshow('before bias', left_ir_image)
        #     cv2.imshow('bias', self.bias_image)
        #
        # left_ir_image = self.overlay_bias_image(left_ir_image)
        #
        # if DEBUG_MODE:
        #     cv2.imshow('after bias', left_ir_image)

        if TRAINING_DATA_COLLECTION_MODE:
            if self.num_frame % 5 == 0:
                # cv2.imwrite('out2/2022-06-10/hover/hover_{}_{}_{}.png'.format(self.saved_image_counter, self.current_exposure, self.current_gain), ir_image_table_1)

                # fake_color = cv2.resize(fake_color, (848 * 3, 480 * 3), interpolation=cv2.INTER_AREA)

                cv2.imwrite('out2/2022-06-30_2/draw2/draw_{}_{}_{}.png'.format(self.saved_image_counter, self.current_exposure, self.current_gain), fake_color)
                print('Saving:', self.saved_image_counter, 'exposure', self.current_exposure, 'gain:', self.current_gain)
                self.saved_image_counter += 1

                # if self.saved_image_counter >= 1000:
                #     print('FINISHED')
                #     sys.exit()

                NUM_SAVE_IMAGES_PER_PERMUTATION = 500
                if self.saved_image_counter >= NUM_SAVE_IMAGES_PER_PERMUTATION:
                    self.saved_image_counter = 0

                    finished, (exposure, gain) = self.get_next_camera_setting_values()
                    if finished:
                        print('FINISHED')
                        sys.exit()
                    self.current_exposure = exposure
                    self.current_gain = gain

                    self.depth_ir_sensor.set_option(rs.option.exposure, exposure)
                    self.depth_ir_sensor.set_option(rs.option.gain, gain)

                    self.depth_ir_sensor_2.set_option(rs.option.exposure, exposure)
                    self.depth_ir_sensor_2.set_option(rs.option.gain, gain)

                    time.sleep(0.1)
        elif CALIBRATION_MODE:
            #print(self.left_ir_image.shape)

            calibration_finished_1 = self.surface_selector.select_surface(left_ir_image_1, CAMERA_PATH_1)
            calibration_finished_2 = self.surface_selector.select_surface(left_ir_image_2, CAMERA_PATH_2)
            # calibration_finished_1 = self.surface_selector_1.select_surface(left_ir_image_1)
            # calibration_finished_2 = self.surface_selector_2.select_surface(left_ir_image_2)
            # calibration_finished = False

            cv2.waitKey(1)

            if calibration_finished_1 and calibration_finished_2:
                print("[Surface Selector Node]: Calibration Finished")
                exit()
        else:
            if DEBUG_MODE:
                pass
                # cv2.imshow('table 0', ir_image_table_0)
                #cv2.imshow('table 1', ir_image_table_1)
                #cv2.imshow('table 2', ir_image_table_2)

                # stacked = cv2.addWeighted(ir_image_table_1, 1, ir_image_table_2, 1, 0)
                # stacked = cv2.resize(stacked, (848*3, 480*3), interpolation = cv2.INTER_AREA)
                # cv2.imshow('stacked', stacked)

                #fake_color = cv2.resize(fake_color, (848 * 3, 480 * 3), interpolation=cv2.INTER_AREA)
                #cv2.imshow('fake color', fake_color)

            # if self.exposure_calibration_mode:
            #     time.sleep(0.1)
            #     max_brightness = np.max(ir_image_table_1)
            #     #print(self.ir_sensor_exposure, max_brightness)
            #     if max_brightness > 240:
            #         if self.ir_sensor_exposure > 50:
            #             self.ir_sensor_exposure -= 50
            #             self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
            #         # if self.ir_sensor_gain > 50:
            #         #     self.ir_sensor_gain -= 50
            #         #     self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
            #         print(f'gain: {self.ir_sensor_gain}, exposure: {self.ir_sensor_exposure}')
            #     elif max_brightness < 200:
            #         if self.ir_sensor_exposure < 10000:
            #             self.ir_sensor_exposure += 50
            #             self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
            #         else:
            #             self.ir_sensor_gain += 50
            #             self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
            #     else:
            #         print(f'exposure: {self.ir_sensor_exposure}, gain: {self.ir_sensor_gain}, max: {np.max(ir_image_table_1)}')
            #         self.exposure_calibration_mode = False
            # elif self.exposure_calibration_mode_2:
            #     mean_brightness = np.mean(ir_image_table_1)
            #     #print(self.ir_sensor_exposure, max_brightness)
            #     if mean_brightness > 45:
            #         if self.ir_sensor_exposure > 50:
            #             self.ir_sensor_exposure -= 50
            #             self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
            #         if self.ir_sensor_gain > 35:
            #             self.ir_sensor_gain -= 50
            #             self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
            #         print(f'gain: {self.ir_sensor_gain}, exposure: {self.ir_sensor_exposure}')
            #     elif mean_brightness < 25:
            #         self.ir_sensor_exposure += 50
            #         self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
            #     else:
            #         print(f'exposure: {self.ir_sensor_exposure}, gain: {self.ir_sensor_gain}, max: {np.max(ir_image_table_1)}')
            #         self.exposure_calibration_mode_2 = False
            #
            # if RECORD_MODE:
            #     # global img_id, write_counter
            #     #write_coutner = 0
            #     cv2.imwrite(f'out/{TRAINING_PATH}/{self.img_id:04d}_{TRAINING_CONDITION}_{self.ir_sensor_exposure}_{self.ir_sensor_gain}.png', ir_image_table_1)
            #     self.ir_sensor_exposure = random.randint(IR_SENSOR_EXPOSURE_MIN, IR_SENSOR_EXPOSURE_MAX)
            #     self.ir_sensor_gain = random.randint(IR_SENSOR_GAIN_MIN, IR_SENSOR_GAIN_MAX)
            #     self.depth_ir_sensor.set_option(rs.option.exposure, self.ir_sensor_exposure)
            #     self.depth_ir_sensor.set_option(rs.option.gain, self.ir_sensor_gain)
            #     time.sleep(0.05)
            #
            #     self.img_id += 1
            #     #time.sleep(1)

            if DEBUG_MODE:
                # img_preview = cv2.cvtColor(ir_image_table, cv2.COLOR_GRAY2BGR)
                # cv2.imshow('test', img_preview)
                cv2.waitKey(1)

    # def find_pen_position_subpixel(self, ir_image, window_name):
    #     _, thresh = cv2.threshold(ir_image, np.max(ir_image) - 1, 255, cv2.THRESH_BINARY)
    #
    #     thresh_large = cv2.resize(thresh, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    #     contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = contours[0] if len(contours) == 2 else contours[1]
    #     min_radius = ir_image.shape[0]
    #     smallest_contour = contours[0]
    #
    #     # print(len(contours))
    #     for contour in contours:
    #         (x, y), radius = cv2.minEnclosingCircle(contour)
    #         if radius < min_radius:
    #             min_radius = radius
    #             smallest_contour = contour
    #
    #     M = cv2.moments(smallest_contour)
    #     # calculate x,y coordinate of center
    #     cX = int(M["m10"] / M["m00"])
    #
    #     cY = int(M["m01"] / M["m00"])
    #
    #     position = (cX, cY)
    #
    #     x_camera_coord = int((cX / 3840) * 848)
    #     y_camera_coord = int((cY / 2160) * 480)
    #
    #     ir_copy = ir_image.copy()
    #     cv2.circle(ir_copy, (x_camera_coord, y_camera_coord), 2, (80))
    #     cv2.imshow('center {}'.format(window_name), ir_copy)
    #     cv2.imshow('thresh', thresh)
    #
    #     return position
    #
    # def find_pen_position(self, ir_image, window_name):
    #     _, thresh = cv2.threshold(ir_image, np.max(ir_image) - 1, 255, cv2.THRESH_BINARY)
    #     contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = contours[0] if len(contours) == 2 else contours[1]
    #     position = (0, 0)
    #     min_radius = ir_image.shape[0]
    #     for contour in contours:
    #         (x, y), radius = cv2.minEnclosingCircle(contour)
    #         if radius < min_radius:
    #             min_radius = radius
    #             position = (int(x), int(y))
    #
    #     ir_copy = ir_image.copy()
    #     cv2.circle(ir_copy, position, 2, (80))
    #     cv2.imshow('center Andi {}'.format(window_name), ir_copy)
    #
    #     return min_radius, position

    # Returns the requested camera frames
    def get_camera_frames(self):
        with self.read_lock:
            if self.new_frames and self.left_ir_image_1 is not None:
                self.new_frames = False
                return self.left_ir_image_1, self.left_ir_image_2, self.matrix_1, self.matrix_2
                # return self.current_ir_image_table_0, self.current_ir_image_table_1, self.current_ir_image_table_2, self.left_ir_image_0, self.left_ir_image_1, self.left_ir_image_2, self.crop_1, self.crop_2
            return None, None, None, None

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



