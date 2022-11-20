import math
import time

import cv2
import numpy as np
from scipy.spatial import distance

MAX_TIME_MISSING_MS = 500
MAX_DIST_BETWEEN_POINTS = 5

SMOOTHING_FACTOR = 0.3

DEBUG_MODE = False

# Relation between border and marker width. Border = PAPER_BORDER_FACTOR * width of marker
PAPER_BORDER_FACTOR = 0.5


class Document:

    def __init__(self, corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right):
        self.corner_top_left = corner_top_left
        self.corner_bottom_left = corner_bottom_left
        self.corner_bottom_right = corner_bottom_right
        self.corner_top_right = corner_top_right

        self.last_seen_timestamp = round(time.time() * 1000)

    def get_document_corner_points(self):
        document_corner_points = [self.corner_top_left[0], self.corner_top_left[1],
                                  self.corner_bottom_left[0], self.corner_bottom_left[1],
                                  self.corner_bottom_right[0], self.corner_bottom_right[1],
                                  self.corner_top_right[0], self.corner_top_right[1]]

        return document_corner_points


class DocumentLocatorService:

    active_document = None

    def __init__(self):
        pass

    def locate_document(self, frame, aruco_markers):

        corner_top_left = None
        corner_bottom_left = None
        corner_bottom_right = None
        corner_top_right = None

        for marker in aruco_markers:

            if marker['id'] == 96:
                vector_x = (marker['corners'][0][0] - marker['corners'][3][0],
                            marker['corners'][0][1] - marker['corners'][3][1])
                vector_y = (marker['corners'][0][0] - marker['corners'][1][0],
                            marker['corners'][0][1] - marker['corners'][1][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][0][0] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[0] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[0]
                y_new = marker['corners'][0][1] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[1] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[1]

                corner_bottom_left = (int(x_new), int(y_new))

            elif marker['id'] == 97:
                vector_x = (marker['corners'][1][0] - marker['corners'][2][0],
                            marker['corners'][1][1] - marker['corners'][2][1])
                vector_y = (marker['corners'][1][0] - marker['corners'][0][0],
                            marker['corners'][1][1] - marker['corners'][0][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][1][0] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    0] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[0]
                y_new = marker['corners'][1][1] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    1] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[1]

                corner_top_left = (int(x_new), int(y_new))

            elif marker['id'] == 98:
                vector_x = (marker['corners'][2][0] - marker['corners'][1][0],
                            marker['corners'][2][1] - marker['corners'][1][1])
                vector_y = (marker['corners'][2][0] - marker['corners'][3][0],
                            marker['corners'][2][1] - marker['corners'][3][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][2][0] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    0] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[0]
                y_new = marker['corners'][2][1] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    1] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[1]

                corner_top_right = (int(x_new), int(y_new))

            elif marker['id'] == 99:
                vector_x = (marker['corners'][3][0] - marker['corners'][0][0],
                            marker['corners'][3][1] - marker['corners'][0][1])
                vector_y = (marker['corners'][3][0] - marker['corners'][2][0],
                            marker['corners'][3][1] - marker['corners'][2][1])
                vector_x_length = self.vector_norm(vector_x)
                vector_y_length = self.vector_norm(vector_y)
                vector_normalized_x = self.normalize_vector(vector_x)
                vector_normalized_y = self.normalize_vector(vector_y)
                x_new = marker['corners'][3][0] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    0] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[0]
                y_new = marker['corners'][3][1] + PAPER_BORDER_FACTOR * vector_x_length * vector_normalized_x[
                    1] + PAPER_BORDER_FACTOR * vector_y_length * vector_normalized_y[1]

                corner_bottom_right = (int(x_new), int(y_new))

        if self.active_document is None:
            if corner_top_left is not None and corner_bottom_left is not None and corner_bottom_right is not None and corner_top_right is not None:
                self.active_document = Document(corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right)

        else:
            self.check_new_pos(corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right)

        if DEBUG_MODE and self.active_document is not None:
            rectangle_points = [np.array([self.active_document.corner_top_left,
                                          self.active_document.corner_bottom_left,
                                          self.active_document.corner_bottom_right,
                                          self.active_document.corner_top_right])]
            cv2.polylines(frame, rectangle_points, isClosed=True, color=(50, 255, 120), thickness=2)
            cv2.imshow('document corners', frame)

        if self.active_document is not None:
            # return self.active_document.contour_points
            return True, self.active_document.get_document_corner_points()
        else:
            return False, []

    def check_new_pos(self, corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right):
        now = round(time.time() * 1000)
        num_found_corners = sum([corner_top_left is not None, corner_bottom_left is not None,
                                 corner_bottom_right is not None, corner_top_right is not None])

        if num_found_corners == 0:
            # Document missing in the current frame
            # If it is missing for a certain timespan, we can assume that it has been removed from the table
            if now - self.active_document.last_seen_timestamp > MAX_TIME_MISSING_MS:
                self.active_document = None
        else:

            if num_found_corners == 4:
                # print('Found all corners')
                self.update_active_document(now, corner_top_left, corner_bottom_left, corner_bottom_right,
                                            corner_top_right)

            elif num_found_corners == 3:
                # print('Found three out of four corners, calculating the missing one:')
                corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right = self.find_single_missing_corner(corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right)
                self.update_active_document(now, corner_top_left, corner_bottom_left, corner_bottom_right,
                                            corner_top_right)

            else:  # Found only one or two corners
                # print('Found one or two corners. Check if they are still in the same place. If yes, no problem')

                has_document_moved = self.has_document_moved(corner_top_left, corner_bottom_left, corner_bottom_right,
                                                             corner_top_right)
                if not has_document_moved:
                    # No prior movement. We can simply use the coordinates from the previous frame
                    if corner_top_left is None:
                        corner_top_left = self.active_document.corner_top_left
                    if corner_bottom_left is None:
                        corner_bottom_left = self.active_document.corner_bottom_left
                    if corner_bottom_right is None:
                        corner_bottom_right = self.active_document.corner_bottom_right
                    if corner_top_right is None:
                        corner_top_right = self.active_document.corner_top_right
                    self.update_active_document(now, corner_top_left, corner_bottom_left, corner_bottom_right,
                                                corner_top_right)
                else:
                    # print('Only one or two corners visible and document has moved')
                    if corner_top_left is not None and corner_top_right is not None:
                        new_corner_bottom_left = self.rotate_point_around_other_point(corner_top_right, corner_top_left, 90)
                        vector_tl_bl = (corner_top_left[0] - new_corner_bottom_left[0],
                                        corner_top_left[1] - new_corner_bottom_left[1])
                        vector_tl_bl_length = self.vector_norm(vector_tl_bl)
                        vector_tl_bl_normalized = self.normalize_vector(vector_tl_bl)

                        corner_bottom_left_new_x = corner_top_left[0] - vector_tl_bl_normalized[0] * (vector_tl_bl_length / 210) * 297
                        corner_bottom_left_new_y = corner_top_left[1] - vector_tl_bl_normalized[1] * (vector_tl_bl_length / 210) * 297
                        new_corner_bottom_left = (int(corner_bottom_left_new_x), int(corner_bottom_left_new_y))

                        new_corner_bottom_right = self.rotate_point_around_other_point(corner_top_left, corner_top_right, 90)
                        vector_tr_br = (corner_top_right[0] - new_corner_bottom_right[0],
                                        corner_top_right[1] - new_corner_bottom_right[1])
                        vector_tr_br_length = self.vector_norm(vector_tr_br)
                        vector_tr_br_normalized = self.normalize_vector(vector_tr_br)

                        corner_bottom_right_new_x = corner_top_right[0] + vector_tr_br_normalized[0] * (
                                    vector_tr_br_length / 210) * 297
                        corner_bottom_right_new_y = corner_top_right[1] + vector_tr_br_normalized[1] * (
                                    vector_tr_br_length / 210) * 297
                        new_corner_bottom_right = (int(corner_bottom_right_new_x), int(corner_bottom_right_new_y))

                        corner_bottom_left = new_corner_bottom_left
                        corner_bottom_right = new_corner_bottom_right

                        self.update_active_document(now, corner_top_left, corner_bottom_left, corner_bottom_right,
                                                    corner_top_right)

    # Checks if the document has been moved since the last frame
    def has_document_moved(self, corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right):
        if corner_top_left is not None:
            if self.has_point_moved(self.active_document.corner_top_left, corner_top_left):
                return True
        if corner_bottom_left is not None:
            if self.has_point_moved(self.active_document.corner_bottom_left, corner_bottom_left):
                return True
        if corner_bottom_right is not None:
            if self.has_point_moved(self.active_document.corner_bottom_right, corner_bottom_right):
                return True
        if corner_top_right is not None:
            if self.has_point_moved(self.active_document.corner_top_right, corner_top_right):
                return True
        return False

    # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python/34374437
    def rotate_point_around_other_point(self, p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    # If we know three out of four document corners, we can directly calculate the missing one
    def find_single_missing_corner(self, corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right):
        if corner_top_left is None:
            corner_top_left = self.__calculate_single_missing_corner(corner_bottom_right, corner_bottom_left,
                                                                     corner_top_right)
        elif corner_bottom_left is None:
            corner_bottom_left = self.__calculate_single_missing_corner(corner_top_right, corner_top_left,
                                                                        corner_bottom_right)
        elif corner_bottom_right is None:
            corner_bottom_right = self.__calculate_single_missing_corner(corner_top_left, corner_top_right,
                                                                         corner_bottom_left)
        else:
            corner_top_right = self.__calculate_single_missing_corner(corner_bottom_left, corner_bottom_right,
                                                                      corner_top_left)

        return corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right

    def __calculate_single_missing_corner(self, existing_corner_one, existing_corner_two, existing_corner_three):
        vector = (existing_corner_one[0] - existing_corner_two[0],
                  existing_corner_one[1] - existing_corner_two[1])
        x_new = existing_corner_three[0] - vector[0]
        y_new = existing_corner_three[1] - vector[1]
        missing_corner = (int(x_new), int(y_new))
        return missing_corner

    def update_active_document(self, now, corner_top_left, corner_bottom_left, corner_bottom_right, corner_top_right):
        self.active_document.last_seen_timestamp = now
        self.active_document.corner_top_left = self.smooth_point(self.active_document.corner_top_left,
                                                                 corner_top_left)
        self.active_document.corner_bottom_left = self.smooth_point(self.active_document.corner_bottom_left,
                                                                    corner_bottom_left)
        self.active_document.corner_bottom_right = self.smooth_point(self.active_document.corner_bottom_right,
                                                                     corner_bottom_right)
        self.active_document.corner_top_right = self.smooth_point(self.active_document.corner_top_right,
                                                                  corner_top_right)
        # self.active_document.contour_points = [corner_top_left[0], corner_top_left[1],
        #                                        corner_bottom_left[0], corner_bottom_left[1],
        #                                        corner_bottom_right[0], corner_bottom_right[1],
        #                                        corner_top_right[0], corner_top_right[1]]

    def smooth_point(self, old_point, new_point):

        MIN_MOVEMENT_THRESHOLD = 3

        if abs(old_point[0] - new_point[0]) < MIN_MOVEMENT_THRESHOLD and abs(old_point[1] - new_point[1]) < MIN_MOVEMENT_THRESHOLD:
            #print('Fixate doc', abs(old_point[0] - new_point[0]), abs(old_point[1] - new_point[1]))
            new_x = old_point[0]
            new_y = old_point[1]
        else:
            #print('update doc pos', abs(old_point[0] - new_point[0]), abs(old_point[1] - new_point[1]))
            new_x = int(SMOOTHING_FACTOR * (new_point[0] - old_point[0]) + old_point[0])
            new_y = int(SMOOTHING_FACTOR * (new_point[1] - old_point[1]) + old_point[1])
        return new_x, new_y

    def has_point_moved(self, old_point, new_point):
        dist = distance.euclidean((old_point[0], old_point[1]), (new_point[0], new_point[1]))
        # print(dist)
        if dist > MAX_DIST_BETWEEN_POINTS:
            return True
        return False

    # From JH - SI Framework
    def dot(self, u, v):
        return sum((a * b) for a, b in zip(u, v))

    # From JH - SI Framework
    def vector_norm(self, v):
        return math.sqrt(self.dot(v, v))

    # From JH - SI Framework
    def normalize_vector(self, v):
        n = float(self.vector_norm(v))
        return [float(v[i]) / n for i in range(len(v))] if n != 0 else [-1 for i in range(len(v))]