#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import datetime


ADAPTIVE_THRESH_CONSTANT = 10  # TODO: Find best value
ARUCO_DICT = aruco.DICT_4X4_100
# ARUCO_DICT = aruco.DICT_6X6_1000

DEBUG_MODE = False
N_FRAMES_DETECTION_RATIO = 100

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

class FiducialsDetectorService:

    corners = []

    aruco_dictionary = None
    aruco_detector_parameters = None
    marker_frame = None

    markers_detected = [False] * N_FRAMES_DETECTION_RATIO
    marker_detection_rate = 0

    def __init__(self):
        self.marker_detection_rate = None
        self.aruco_dictionary = aruco.Dictionary_get(ARUCO_DICT)
        self.aruco_detector_parameters = aruco.DetectorParameters_create()
        print('[FiducialsDetectorService]: Ready')

    #@timeit("detection")
    def detect_fiducials(self, frame):
        # If color image, convert to grey

        # print(frame.shape)

        corners, ids, rejected_points = aruco.detectMarkers(frame, self.aruco_dictionary,
                                                            parameters=self.aruco_detector_parameters)

        aruco_markers = []

        if DEBUG_MODE:
            preview = frame.copy()
            aruco.drawDetectedMarkers(preview, corners, ids)  # draw a square around the markers
            cv2.imshow('marker', preview)

        # check if the list of IDs is not empty
        if np.all(ids is not None):
            for i in range(len(ids)):
                aruco_marker = {'id': ids[i][0],
                                'angle': self.calculate_aruco_marker_rotation(corners[i][0], frame),
                                'corners': corners[i][0],
                                'centroid': self.centroid(corners[i][0])}
                aruco_markers.append(aruco_marker)

        self.markers_detected = self.markers_detected[1:N_FRAMES_DETECTION_RATIO] + [True if (len(aruco_markers) > 0) else False]
        self.marker_detection_rate = sum(self.markers_detected)/N_FRAMES_DETECTION_RATIO

        return aruco_markers


    # Calculate the rotation_diff of an aruco marker relative to the frame.
    # Returns the angle in the range from 0° to 360°
    def calculate_aruco_marker_rotation(self, aruco_marker_corners, frame):
        tracker_point_one = aruco_marker_corners[0]
        tracker_point_two = aruco_marker_corners[1]
        vector_one = np.array([frame.shape[0], 0])
        vector_two = np.array([tracker_point_two[0] - tracker_point_one[0], tracker_point_two[1] - tracker_point_one[1]])

        angle = self.calculate_angle(vector_one, vector_two)
        return angle

    # Calculate the angle between the two given vectors
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    def calculate_angle(self, v1, v2):
        angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angle = np.degrees(angle)
        if angle < 0:
            angle = angle + 360
        return int(angle)

    # Get the centroid of a polygon
    # https://progr.interplanety.org/en/python-how-to-find-the-polygon-center-coordinates/
    def centroid(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = sum(_x_list) / _len
        _y = sum(_y_list) / _len
        return _x, _y
