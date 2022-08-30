#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import numpy as np
import configparser

CONFIG_FILE_NAME = 'config.ini'

FLIP_IMAGE = False

class TableExtractionService:
    config = {}

    # TODO: reload config file
    def __init__(self):
        print('init extractor')
        self.read_config_file()

    # In the config file, info like the table corner coordinates are stored
    def read_config_file(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_NAME)

        if len(config.sections()) > 0:
            for section in config.sections():
                self.config[section] = {}
                for key, value in config.items(section):
                    self.config[section][key] = eval(value)
        else:
            print('[TableExtractionService]: Could not find calibration info. Set camera to calibration mode and try again')
            sys.exit(1)

    # TODO: Check differences between camera and table aspect ratio
    # Based on: https://www.youtube.com/watch?v=PtCQH93GucA
    def extract_table_area(self, frame, camera_parameter_name):
        if frame is None:
            return None

        x = frame.shape[1]
        y = frame.shape[0]

        if camera_parameter_name not in self.config.keys():
            # print('No calibration for', camera_parameter_name)
            return None

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


        # if FLIP_IMAGE:
        #     pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [0, y], [x, 0], [0, 0]])

        # if FLIP_IMAGE:
        #     pts2 = np.float32([[int(-1.2 * x), int(-1.2 * y)], [x, int(1.2 * y)], [int(1.2 * x), y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [int(-1.2 * x), y], [x, int(-1.2 * y)], [int(-1.2 * x), int(-1.2 * y)]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # print(matrix)

        frame = cv2.warpPerspective(frame, matrix, (x, y))
        return frame

        # TODO: Check differences between camera and table aspect ratio
        # Based on: https://www.youtube.com/watch?v=PtCQH93GucA

    def extract_table_area_wide(self, frame, camera_parameter_name):
        if frame is None:
            return None

        x = frame.shape[1]
        y = frame.shape[0]

        if camera_parameter_name not in self.config.keys():
            # print('No calibration for', camera_parameter_name)
            return None

        pts1 = np.float32([self.config[camera_parameter_name]['cornertopleft'],
                           self.config[camera_parameter_name]['cornertopright'],
                           self.config[camera_parameter_name]['cornerbottomleft'],
                           self.config[camera_parameter_name]['cornerbottomright']])

        x0 = int(0.05 * x)
        y0 = int(0.05 * y)
        x1 = int(0.95 * x)
        y1 = int(0.95 * y)

        #print('coords', x0, y0, x1, y1)

        if FLIP_IMAGE:
            pts2 = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        else:
            pts2 = np.float32([[x1, y1], [x0, y1], [x1, y0], [y0, y0]])

        # if FLIP_IMAGE:
        #     pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [0, y], [x, 0], [0, 0]])

        # if FLIP_IMAGE:
        #     pts2 = np.float32([[int(-1.2 * x), int(-1.2 * y)], [x, int(1.2 * y)], [int(1.2 * x), y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [int(-1.2 * x), y], [x, int(-1.2 * y)], [int(-1.2 * x), int(-1.2 * y)]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (x, y))
        return frame, (x0, y0, x1, y1)

    def get_homography(self, width, height, camera_parameter_name):
        x = width
        y = height

        if camera_parameter_name not in self.config.keys():
            # print('No calibration for', camera_parameter_name)
            return None

        pts1 = np.float32([self.config[camera_parameter_name]['cornertopleft'],
                           self.config[camera_parameter_name]['cornertopright'],
                           self.config[camera_parameter_name]['cornerbottomleft'],
                           self.config[camera_parameter_name]['cornerbottomright']])

        x0 = 0
        y0 = 0
        x1 = x
        y1 = y

        #print('coords', x0, y0, x1, y1)

        if FLIP_IMAGE:
            pts2 = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        else:
            pts2 = np.float32([[x1, y1], [x0, y1], [x1, y0], [y0, y0]])

        # if FLIP_IMAGE:
        #     pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [0, y], [x, 0], [0, 0]])

        # if FLIP_IMAGE:
        #     pts2 = np.float32([[int(-1.2 * x), int(-1.2 * y)], [x, int(1.2 * y)], [int(1.2 * x), y], [x, y]])
        # else:
        #     pts2 = np.float32([[x, y], [int(-1.2 * x), y], [x, int(-1.2 * y)], [int(-1.2 * x), int(-1.2 * y)]])

        homography, status = cv2.findHomography(pts1, pts2)
        # print(homography)
        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # print(matrix)

        return homography

