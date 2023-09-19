
import os
import cv2
import numpy as np
import configparser

CONFIG_FILE_PATH = os.path.join(os.getcwd(), 'TipTrack', 'config')
CONFIG_FILE_NAME = 'config.ini'

FLIP_IMAGE = False  # Flip the output image 180° -> Needed if cameras see the projection area upside down


class SurfaceExtractor:
    """ SurfaceExtractor

        This script allows you to extract a rectangular area from a camera frame based on coordinates in a config file,
        of just get the homography values if you want to do this transformation by yourself later on.

    """
    config = {}
    config_available = False

    # TODO: reload config file
    def __init__(self):
        self.read_config_file()

    # In the config file, info like the table corner coordinates are stored
    def read_config_file(self):
        if not os.path.exists(CONFIG_FILE_PATH):
            print('[SurfaceExtractor]: No config file found')
        else:

            config = configparser.ConfigParser()
            config.read(os.path.join(CONFIG_FILE_PATH, CONFIG_FILE_NAME))

            if len(config.sections()) > 0:
                for section in config.sections():
                    self.config[section] = {}
                    for key, value in config.items(section):
                        self.config[section][key] = eval(value)

                self.config_available = True
            else:
                print('[SurfaceExtractor]: Config file found, but no calibration info')
                # sys.exit(1)

    # TODO: Check differences between camera and table aspect ratio
    # Based on: https://www.youtube.com/watch?v=PtCQH93GucA
    def extract_table_area(self, frame, camera_parameter_name):
        if frame is None or camera_parameter_name not in self.config.keys():
            return None

        x = frame.shape[1]
        y = frame.shape[0]

        pts1, pts2 = self.__get_points(camera_parameter_name, x, y)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (x, y))
        return frame

    def get_homography(self, width, height, camera_parameter_name):

        if not self.config_available:
            return None

        if camera_parameter_name not in self.config.keys():
            return None

        pts1, pts2 = self.__get_points(camera_parameter_name, width, height)

        homography, status = cv2.findHomography(pts1, pts2)

        return homography

    def __get_points(self, camera_parameter_name, x, y):
        pts1 = np.float32([self.config[camera_parameter_name]['cornertopleft'],
                           self.config[camera_parameter_name]['cornertopright'],
                           self.config[camera_parameter_name]['cornerbottomleft'],
                           self.config[camera_parameter_name]['cornerbottomright']])

        x0 = 0
        y0 = 0
        x1 = x
        y1 = y

        if FLIP_IMAGE:
            pts2 = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        else:
            pts2 = np.float32([[x1, y1], [x0, y1], [x1, y0], [y0, y0]])

        return pts1, pts2

