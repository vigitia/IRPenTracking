#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import configparser

CONFIG_FILE_NAME = 'config.ini'


class TableExtractionService:
    config = {}

    #TODO: reload config file
    def __init__(self):
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

            # print(self.config)
            # print('[TableExtractionService]: Successfully read data from config file')
        else:
            print('[TableExtractionService]: Could not find calibration info')

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

        pts2 = np.float32([[x, y], [0, y], [x, 0], [0, 0]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        frame = cv2.warpPerspective(frame, matrix, (x, y))
        return frame

    # def get_table_border(self):
    #     table_border = np.array([self.table_corner_top_left, self.table_corner_top_right,
    #                              self.table_corner_bottom_right, self.table_corner_bottom_left])
    #
    #     return table_border
