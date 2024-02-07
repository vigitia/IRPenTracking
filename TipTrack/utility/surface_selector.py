#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path

import cv2
import configparser

# https://stackoverflow.com/questions/9763116/parse-a-tuple-from-a-string
from ast import literal_eval as make_tuple  # Needed to convert strings stored in config file back to tuples

CIRCLE_DIAMETER = 2
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_COLOR_OLD = (0, 255, 0)
FONT_COLOR = (0, 0, 0)

CONFIG_FILE_PATH = os.path.join(os.getcwd(), 'TipTrack', 'config')
# CONFIG_FILE_PATH = os.path.join('../../', 'TipTrack', 'config')
CONFIG_FILE_NAME = 'config.ini'


RESCALE = True


class SurfaceSelector:
    """ SurfaceSelector

        This class allows you to select the corners of the projection surface.
        The camera feeds will be shown in preview windows. Simply click on the for corners using your mouse.

    """

    last_mouse_click_coordinates = {}

    calibration_data = {}

    def __init__(self):
        self.read_config_file()

    # Log mouse click positions to the console
    def on_mouse_click(self, event, x, y, flags, window_name):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_mouse_click_coordinates[window_name].append((x, y))

    # In the config file, info like the table corner coordinates are stored
    def read_config_file(self):

        config = configparser.ConfigParser()

        if not os.path.exists(CONFIG_FILE_PATH):
            print('No config file found, creating one...')
            config.write(open(os.path.join(CONFIG_FILE_PATH, CONFIG_FILE_NAME), 'w'))

        config.read(os.path.join(CONFIG_FILE_PATH, CONFIG_FILE_NAME))

        print('Sections in config file:', config.sections())

        if len(config.sections()) > 0:
            for section in config.sections():
                self.calibration_data[section] = {
                    'corner_top_left': make_tuple(config[section]['CornerTopLeft']),
                    'corner_top_right': make_tuple(config[section]['CornerTopRight']),
                    'corner_bottom_left': make_tuple(config[section]['CornerBottomLeft']),
                    'corner_bottom_right': make_tuple(config[section]['CornerBottomRight'])
                }

            print('[Calibration Mode]: Successfully read data from config file')
        # else:
        #     print('[Calibration Mode]: Error reading data from config file')

    def select_surface(self, frame, camera_parameter_name):
        calibration_finished = False

        if frame is not None:
            calibration_finished = self.display_mode_calibration(frame, camera_parameter_name)
        else:
            print("[Calibration Mode]: Please wait until camera is ready...")

        return calibration_finished

    def __display_previous_calibration(self, frame, camera_parameter_name):

        if camera_parameter_name in self.calibration_data.keys():
            # Display previous points
            cv2.circle(frame, self.calibration_data[camera_parameter_name]['corner_top_left'],
                       CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
            cv2.circle(frame, self.calibration_data[camera_parameter_name]['corner_top_right'],
                       CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
            cv2.circle(frame, self.calibration_data[camera_parameter_name]['corner_bottom_left'],
                       CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)
            cv2.circle(frame, self.calibration_data[camera_parameter_name]['corner_bottom_right'],
                       CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, 1)

        return frame

    def display_mode_calibration(self, frame, camera_parameter_name):

        if RESCALE:
            frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = self.__display_previous_calibration(frame, camera_parameter_name)

        if not camera_parameter_name in self.last_mouse_click_coordinates.keys():
            self.last_mouse_click_coordinates[camera_parameter_name] = []

        # Draw circles for clicks in a different color to mark the new points
        for coordinate in self.last_mouse_click_coordinates[camera_parameter_name]:
            cv2.circle(frame, coordinate, CIRCLE_DIAMETER, CIRCLE_COLOR, 1)

        if len(self.last_mouse_click_coordinates[camera_parameter_name]) == 4:
            print('[Calibration Mode]: Calibrated camera', camera_parameter_name)
            self.update_table_corner_calibration(self.last_mouse_click_coordinates[camera_parameter_name], camera_parameter_name)
            cv2.destroyWindow(camera_parameter_name)
            return True
        else:
            cv2.imshow(camera_parameter_name, frame)
            cv2.setMouseCallback(camera_parameter_name, self.on_mouse_click, camera_parameter_name)
            cv2.waitKey(1)
            return False

    def update_table_corner_calibration(self, coordinates, camera_parameter_name):
        # Order coordinates by x value
        coordinates = sorted(coordinates)

        if RESCALE:
            for i, coordinate in enumerate(coordinates):
                coordinates[i] = (coordinate[0] * 2, coordinate[1] * 2)

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
        config.read(os.path.join(CONFIG_FILE_PATH, CONFIG_FILE_NAME))

        if camera_parameter_name not in config.sections():
            config.add_section(camera_parameter_name)

        config.set(camera_parameter_name, 'CornerTopLeft', str(self.calibration_data['corner_top_left']))
        config.set(camera_parameter_name, 'CornerTopRight', str(self.calibration_data['corner_top_right']))
        config.set(camera_parameter_name, 'CornerBottomLeft', str(self.calibration_data['corner_bottom_left']))
        config.set(camera_parameter_name, 'CornerBottomRight', str(self.calibration_data['corner_bottom_right']))

        with open(os.path.join(CONFIG_FILE_PATH, CONFIG_FILE_NAME), 'w') as configfile:
            config.write(configfile)
