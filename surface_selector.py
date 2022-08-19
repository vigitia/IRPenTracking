#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import configparser

# https://stackoverflow.com/questions/9763116/parse-a-tuple-from-a-string
from ast import literal_eval as make_tuple  # Needed to convert strings stored in config file back to tuples

CIRCLE_DIAMETER = 2
CIRCLE_COLOR_OLD = (0, 0, 255)
CIRCLE_COLOR_NEW = (0, 255, 0)
FONT_COLOR = (0, 0, 255)
HINT = 'Click on each of the four corners of the projection'

CONFIG_FILE_NAME = 'config.ini'

SHOW_OLD_POSITIONS = False


class SurfaceSelector:
    """ SurfaceSelector

        This class allows you to select the corners of the projection surface using a simple GUI.
        This will be needed to rectify the camera images, extract the table area and calibrate the system.

    """

    last_mouse_click_coordinates = {}

    # TODO: Add support for differently shaped tables (not just rectangles)
    table_corner_top_left = (0, 0)
    table_corner_top_right = (0, 0)
    table_corner_bottom_left = (0, 0)
    table_corner_bottom_right = (0, 0)

    calibration_data = {}

    def __init__(self):
        # self.camera_parameter_name = camera_parameter_name

        self.read_config_file()
        # self.init_opencv()

    # def init_opencv(self):
    #     #cv2.startWindowThread()
    #     #cv2.namedWindow('Surface Selector', cv2.WINDOW_AUTOSIZE)
    #
    #     # Set mouse callbacks to extract the coordinates of clicked spots in the roi
    #     #cv2.setMouseCallback('Surface Selector', self.on_mouse_click)
    #     pass

    # Log mouse click positions to the console
    def on_mouse_click(self, event, x, y, flags, window_name):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print((x, y))
            self.last_mouse_click_coordinates[window_name].append((x, y))
            # Reset list after four clicks
            # if len(self.last_mouse_click_coordinates[window_name]) > 4:
            #     self.last_mouse_click_coordinates[window_name] = []

    # In the config file, info like the table corner coordinates are stored
    def read_config_file(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_NAME)
        print('Sections in config file:', config.sections())

        if len(config.sections()) > 0:
            for section in config.sections():
                self.calibration_data[section] = {
                    'corner_top_left': make_tuple(config[section]['CornerTopLeft']),
                    'corner_top_right': make_tuple(config[section]['CornerTopRight']),
                    'corner_bottom_left': make_tuple(config[section]['CornerBottomLeft']),
                    'corner_bottom_right': make_tuple(config[section]['CornerBottomRight'])
                }

            # try:
            #     # Coordinates of table corners for perspective transformation
            #     self.table_corner_top_left = make_tuple(config[self.camera_parameter_name]['CornerTopLeft'])
            #     self.table_corner_top_right = make_tuple(config[self.camera_parameter_name]['CornerTopRight'])
            #     self.table_corner_bottom_left = make_tuple(config[self.camera_parameter_name]['CornerBottomLeft'])
            #     self.table_corner_bottom_right = make_tuple(config[self.camera_parameter_name]['CornerBottomRight'])
            # except KeyError as e:
            #     print(e)

            print('[Calibration Mode]: Successfully read data from config file')
        else:
            print('[Calibration Mode]: Error reading data from config file')
            # config.add_section(self.camera_parameter_name)
            #with open(CONFIG_FILE_NAME, 'w') as configfile:
            #    config.write(configfile)

    def select_surface(self, frame, camera_parameter_name):
        calibration_finished = False

        if frame is not None:
            calibration_finished = self.display_mode_calibration(frame, camera_parameter_name)
        else:
            print("[Calibration Mode]: Please wait until camera is ready...")

        # print(self.calibration_data)
        
        return calibration_finished

    def display_mode_calibration(self, frame, camera_parameter_name):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # if SHOW_OLD_POSITIONS:
        #     # Show circles of previous coordinates
        #     cv2.circle(frame, self.table_corner_top_left, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
        #     cv2.circle(frame, self.table_corner_top_right, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
        #     cv2.circle(frame, self.table_corner_bottom_left, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
        #     cv2.circle(frame, self.table_corner_bottom_right, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)

        if not camera_parameter_name in self.last_mouse_click_coordinates.keys():
            self.last_mouse_click_coordinates[camera_parameter_name] = []

        # Draw circles for clicks in a different color to mark the new points
        for coordinate in self.last_mouse_click_coordinates[camera_parameter_name]:
            cv2.circle(frame, coordinate, CIRCLE_DIAMETER, CIRCLE_COLOR_NEW, 1)

        # Add text to explain what to do
        # cv2.putText(img=frame, text=HINT + ' (' + str(len(self.last_mouse_click_coordinates)) + '/4)',
        #             org=(int(frame.shape[1] / 6), int(frame.shape[0] / 8)),
        #             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 255))

        if len(self.last_mouse_click_coordinates[camera_parameter_name]) == 4:
            print('[Calibration Mode]: Calibrated camera', camera_parameter_name)
            self.update_table_corner_calibration(self.last_mouse_click_coordinates[camera_parameter_name], camera_parameter_name)
            cv2.destroyWindow(camera_parameter_name)
            return True
        else:
            cv2.imshow(camera_parameter_name, frame)
            cv2.setMouseCallback(camera_parameter_name, self.on_mouse_click, camera_parameter_name)
            #cv2.imshow('test', frame)
            #print('dmc end')
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
            return False

    def update_table_corner_calibration(self, coordinates, camera_parameter_name):
        # Order coordinates by x value
        coordinates = sorted(coordinates)

        if coordinates[0][1] > coordinates[1][1]:
            self.calibration_data['corner_top_left'] = coordinates[1]
            self.calibration_data['corner_bottom_left'] = coordinates[0]
        else:
            self.calibration_data['corner_top_left'] = coordinates[0]
            self.calibration_data['corner_bottom_left'] = coordinates[1]

        if coordinates[2][1] > coordinates[3][1]:
            self.calibration_data['corner_top_right'] = coordinates[3]
            self.calibration_data['corner_bottom_right'] = coordinates[2]
        else:
            self.calibration_data['corner_top_right'] = coordinates[2]
            self.calibration_data['corner_bottom_right'] = coordinates[3]

        # Update config
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_NAME)

        if camera_parameter_name not in config.sections():
            config.add_section(camera_parameter_name)

        config.set(camera_parameter_name, 'CornerTopLeft', str(self.calibration_data['corner_top_left']))
        config.set(camera_parameter_name, 'CornerTopRight', str(self.calibration_data['corner_top_right']))
        config.set(camera_parameter_name, 'CornerBottomLeft', str(self.calibration_data['corner_bottom_left']))
        config.set(camera_parameter_name, 'CornerBottomRight', str(self.calibration_data['corner_bottom_right']))

        with open(CONFIG_FILE_NAME, 'w') as configfile:
            config.write(configfile)

    # def dot(self, u, v):
    #     return sum((a * b) for a, b in zip(u, v))
    #
    # def vector_norm(self, v):
    #     return math.sqrt(self.dot(v, v))
    #
    # def normalize_vector(self, v):
    #     n = float(self.vector_norm(v))
    #     return [float(v[i]) / n for i in range(len(v))] if n != 0 else [-1 for i in range(len(v))]
