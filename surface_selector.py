#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import configparser

# https://stackoverflow.com/questions/9763116/parse-a-tuple-from-a-string
from ast import literal_eval as make_tuple  # Needed to convert strings stored in config file back to tuples

CIRCLE_DIAMETER = 5
CIRCLE_COLOR_OLD = (0, 0, 255)
CIRCLE_COLOR_NEW = (0, 255, 0)
FONT_COLOR = (0, 0, 255)
HINT = 'Click on each of the four corners of the table/projection'

CONFIG_FILE_NAME = 'config.ini'

AUTO_CALIBRATION = False


class SurfaceSelector:
    """ SurfaceSelector

        This class allows you to select the corners of the table using a simple GUI.
        This will be needed to rectify the camera images, extract the table area and calibrate the system.

    """

    last_mouse_click_coordinates = []

    # TODO: Add support for differently shaped tables (not just rectangles)
    table_corner_top_left = (0, 0)
    table_corner_top_right = (0, 0)
    table_corner_bottom_left = (0, 0)
    table_corner_bottom_right = (0, 0)

    def __init__(self, camera_parameter_name):
        self.camera_parameter_name = camera_parameter_name

        self.read_config_file()
        self.init_opencv()

    def init_opencv(self):
        #cv2.startWindowThread()
        #cv2.namedWindow('Surface Selector', cv2.WINDOW_AUTOSIZE)

        # Set mouse callbacks to extract the coordinates of clicked spots in the image
        #cv2.setMouseCallback('Surface Selector', self.on_mouse_click)
        pass

    # Log mouse click positions to the console
    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print((x, y))
            self.last_mouse_click_coordinates.append((x, y))
            # Reset list after four clicks
            if len(self.last_mouse_click_coordinates) > 4:
                self.last_mouse_click_coordinates = []

    # In the config file, info like the table corner coordinates are stored
    def read_config_file(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_NAME)
        print(config.sections())

        if len(config.sections()) > 0:

            try:
                # Coordinates of table corners for perspective transformation
                self.table_corner_top_left = make_tuple(config[self.camera_parameter_name]['CornerTopLeft'])
                self.table_corner_top_right = make_tuple(config[self.camera_parameter_name]['CornerTopRight'])
                self.table_corner_bottom_left = make_tuple(config[self.camera_parameter_name]['CornerBottomLeft'])
                self.table_corner_bottom_right = make_tuple(config[self.camera_parameter_name]['CornerBottomRight'])
            except KeyError as e:
                print(e)

            print('[Calibration Mode]: Successfully read data from config file')
        else:
            print('[Calibration Mode]: Error reading data from config file')
            config.add_section(self.camera_parameter_name)
            with open(CONFIG_FILE_NAME, 'w') as configfile:
                config.write(configfile)

    def select_surface(self, frame):
        if frame is not None:
            if AUTO_CALIBRATION:
                calibration_finished = self.auto_calibrate(frame)
            else:
                print('select surface')
                calibration_finished = self.display_mode_calibration(frame)
                print('select surface end')
        else:
            print("[Calibration Mode]: Please wait until camera is ready...")
        
        return calibration_finished
        
        # key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()

    def __find_corners_by_aruco_markers(self, color_image):
        aruco_markers = self.fiducials_detection_service.detect_fiducials(color_image)
        # aruco.drawDetectedMarkers(preview, corners, ids)

        corners = []

        for marker in aruco_markers:

            marker_id = marker['id']

            if marker_id == 0:
                vector_x = (marker['corners'][0][0] - marker['corners'][1][0],
                            marker['corners'][0][1] - marker['corners'][1][1])
                vector_y = (marker['corners'][0][0] - marker['corners'][3][0],
                            marker['corners'][0][1] - marker['corners'][3][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][0][0] + vector_x_length * vector_normalized_x[0] + vector_y_length * \
                        vector_normalized_y[0]
                y_new = marker['corners'][0][1] + vector_x_length * vector_normalized_x[1] + vector_y_length * \
                        vector_normalized_y[1]

                corner_one = (int(x_new), int(y_new))
                color_image = cv2.circle(color_image, corner_one, 2, (255, 255, 0), -1)
                corners.append(corner_one)
            elif marker_id == 1:
                vector_x = (marker['corners'][3][0] - marker['corners'][2][0],
                            marker['corners'][3][1] - marker['corners'][2][1])
                vector_y = (marker['corners'][3][0] - marker['corners'][0][0],
                            marker['corners'][3][1] - marker['corners'][0][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][3][0] + vector_x_length * vector_normalized_x[0] + vector_y_length * \
                        vector_normalized_y[0]
                y_new = marker['corners'][3][1] + vector_x_length * vector_normalized_x[1] + vector_y_length * \
                        vector_normalized_y[1]

                corner_two = (int(x_new), int(y_new))
                color_image = cv2.circle(color_image, corner_two, 2, (255, 255, 0), -1)
                corners.append(corner_two)
            elif marker_id == 2:
                vector_x = (marker['corners'][2][0] - marker['corners'][3][0],
                            marker['corners'][2][1] - marker['corners'][3][1])
                vector_y = (marker['corners'][2][0] - marker['corners'][1][0],
                            marker['corners'][2][1] - marker['corners'][1][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][2][0] + vector_x_length * vector_normalized_x[0] + vector_y_length * \
                        vector_normalized_y[0]
                y_new = marker['corners'][2][1] + vector_x_length * vector_normalized_x[1] + vector_y_length * \
                        vector_normalized_y[1]

                corner_three = (int(x_new), int(y_new))
                color_image = cv2.circle(color_image, corner_three, 2, (255, 255, 0), -1)
                corners.append(corner_three)
            elif marker_id == 3:
                vector_x = (marker['corners'][1][0] - marker['corners'][0][0],
                            marker['corners'][1][1] - marker['corners'][0][1])
                vector_y = (marker['corners'][1][0] - marker['corners'][2][0],
                            marker['corners'][1][1] - marker['corners'][2][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][1][0] + vector_x_length * vector_normalized_x[0] + vector_y_length * \
                        vector_normalized_y[0]
                y_new = marker['corners'][1][1] + vector_x_length * vector_normalized_x[1] + vector_y_length * \
                        vector_normalized_y[1]

                corner_four = (int(x_new), int(y_new))
                color_image = cv2.circle(color_image, corner_four, 2, (255, 255, 0), -1)
                corners.append(corner_four)

        return color_image, corners

    def auto_calibrate(self, frame):
        cv2.imshow('Surface Selector', frame)

        frame, corners = self.__find_corners_by_aruco_markers(frame)

        if len(corners) == 4:
            print('[Calibration Mode]: Calibrated')
            self.update_table_corner_calibration(corners)
            return True
        return False

    def display_mode_calibration(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Show circles of previous coordinates
        cv2.circle(frame, self.table_corner_top_left, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, -1)
        cv2.circle(frame, self.table_corner_top_right, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, -1)
        cv2.circle(frame, self.table_corner_bottom_left, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, -1)
        cv2.circle(frame, self.table_corner_bottom_right, CIRCLE_DIAMETER, CIRCLE_COLOR_OLD, -1)

        # Draw circles for clicks in a different color to mark the new points
        for coordinate in self.last_mouse_click_coordinates:
            cv2.circle(frame, coordinate, CIRCLE_DIAMETER, CIRCLE_COLOR_NEW, -1)

        # Add text to explain what to do
        cv2.putText(img=frame, text=HINT + ' (' + str(len(self.last_mouse_click_coordinates)) + '/4)',
                    org=(int(frame.shape[1] / 6), int(frame.shape[0] / 8)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 255))

        if len(self.last_mouse_click_coordinates) == 4:
            print('[Calibration Mode]: Calibrated')
            self.update_table_corner_calibration(self.last_mouse_click_coordinates)
            cv2.destroyAllWindows()
            return True

        print('dmc imshow')
        cv2.imshow('Surface Selector', frame)
        cv2.setMouseCallback('Surface Selector', self.on_mouse_click)
        #cv2.imshow('test', frame)
        #print('dmc end')
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()
        return False

    def update_table_corner_calibration(self, coordinates):
        # Order coordinates by x value
        coordinates = sorted(coordinates)

        if coordinates[0][1] > coordinates[1][1]:
            self.table_corner_top_left = coordinates[1]
            self.table_corner_bottom_left = coordinates[0]
        else:
            self.table_corner_top_left = coordinates[0]
            self.table_corner_bottom_left = coordinates[1]

        if coordinates[2][1] > coordinates[3][1]:
            self.table_corner_top_right = coordinates[3]
            self.table_corner_bottom_right = coordinates[2]
        else:
            self.table_corner_top_right = coordinates[2]
            self.table_corner_bottom_right = coordinates[3]

        # Update config
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_NAME)

        if self.camera_parameter_name not in config.sections():
            config.add_section(self.camera_parameter_name)

        config.set(self.camera_parameter_name, 'CornerTopLeft', str(self.table_corner_top_left))
        config.set(self.camera_parameter_name, 'CornerTopRight', str(self.table_corner_top_right))
        config.set(self.camera_parameter_name, 'CornerBottomLeft', str(self.table_corner_bottom_left))
        config.set(self.camera_parameter_name, 'CornerBottomRight', str(self.table_corner_bottom_right))

        with open(CONFIG_FILE_NAME, 'w') as configfile:
            config.write(configfile)

    def project_targets_on_table(self):
        # TODO: Project Targets on table and let a user click them.
        pass

    def dot(self, u, v):
        return sum((a * b) for a, b in zip(u, v))

    def vector_norm(self, v):
        return math.sqrt(self.dot(v, v))

    def normalize_vector(self, v):
        n = float(self.vector_norm(v))
        return [float(v[i]) / n for i in range(len(v))] if n != 0 else [-1 for i in range(len(v))]
