#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import statistics
import time
from enum import Enum

import cv2
import numpy as np

from scipy.spatial import distance

from colorama import Fore, Back, Style

# from pen_state_svm import PenStateSVM
from pen_state_smp import PenStateSMP

# Minimum and maximum Radius of a Spot for to count
MIN_PEN_SPOT_RADIUS = 1
MAX_PEN_SPOT_RADIUS = 10

MIN_DISTANCE_BETWEEN_POINTS = 4 * MAX_PEN_SPOT_RADIUS

# Min brightness value (from 0 to 255) the
BRIGHTNESS_CUTOFF_THRESHOLD = 100

# Amount of time a point can be missing until the event "on click/drag stop" will be fired
TIME_POINT_MISSING_THRESHOLD_MS = 50

# Max distance in pixel a point between to frames can be until it will be treated as a new event
DISTANCE_MOVEMENT_BETWEEN_FRAMES_THRESHOLD = 800

# Point needs to appear and disappear within this timeframe in ms to count as a click (vs. a drag event)
CLICK_THRESH_MS = 250


# No need to allow for more points than there are Laser Pointer Pens
MAX_NUM_POINTS = 3

# TODO: Check if this works:
MIN_TIME_EXISTING_MS = 50  # Ignore all points that appear only for an unrealistic amount of time

# Simple Smoothing
SMOOTHING_FACTOR = 0.9  # Value between 0 and 1, depending if the old or the new value should count more.

DEBUG_MODE = False

# Set to True if you use a Laser Pointer instead of an IR Light Pen
USE_LASER_POINTER = False

CALIBRATION_FILE = ["touch.csv", "hover.csv", "away.csv"]


# DONT_CHECK_FOR_COLOR = True
#
# # lower HSV mask for red(0-10)
# LOWER_RED_BOTTOM_RANGE = np.array([0, 70, 50])
# UPPER_RED_BOTTOM_RANGE = np.array([10, 255, 255])
#
# # upper HSV mask for red(170-180)
# LOWER_RED_TOP_RANGE = np.array([170, 70, 50])
# UPPER_RED_TOP_RANGE = np.array([180, 255, 255])
#
# LOWER_SPOT = np.array([101, 23, 185])
# UPPER_SPOT = np.array([180, 90, 255])
#
# # At least X% of pixels around a detected spot need to be red to be counted as a laser pointer point
# MIN_PERCENTAGE_RED = 2


# State of a Point
class State(Enum):
    NEW = -1  # A new event where it is not yet sure if it will just be a click event or a drag event
    CLICK = 0
    DRAG = 1
    DOUBLE_CLICK = 2
    HOVER = 3
    RUBBING = 4


class Point:

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.id = -1

        self.missing = False
        self.last_seen_timestamp = 0

        self.first_appearance = round(time.time() * 1000)
        self.state = State.NEW
        self.history = [(x, y)]

        self.alive = True

    def get_coordinates(self):
        return tuple([self.x, self.y])

    def __repr__(self):
        return 'IR Point {} at ({}, {}). Type: {}. Num Points: {}'.format(str(self.id), str(self.x), str(self.y), self.state, len(self.history))


class LaserPenDetectorService:
    """ LaserPenDetectorService

        Uses an IR camera image to find bright spots caused by an IR emitting light source
        (in this case a red laser pointer)

    """

    active_points = []
    highest_id = 1
    stored_lines = []
    points_to_remove = []  # Points that got deleted from active_points in the current frame (for alive message)
    double_click_candidates = []

    selected_color = [0, 255, 0]

    current_color_image = None

    # Collect all new lines here. Will be emptied every frame
    new_lines = []

    svm = None
    smp = None

    def __init__(self):
        print('[LaserPointerDetectionService]: Ready')

    # Find bright spots in the IR image. If their size and brightness matches, we can assume that they are from the
    # laser pointer

    def reset_color(self):
        self.selected_color = [0, 255, 0]

    def collect_training_data(self, ir_frame):
        thresh = cv2.threshold(ir_frame, BRIGHTNESS_CUTOFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        if DEBUG_MODE:
            cv2.imshow('tresh', thresh)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            return self.detect_ir_points(largest_contour, ir_frame, collect_training_data=True)

        return None

    def get_pen_spots(self, color_image, ir_frame, change_color):

        # TODO: REMOVE this line
        change_color = False

        if self.svm is None:
            self.svm = PenStateSVM.create(CALIBRATION_FILE)

        if self.smp is None:
            self.smp = PenStateSMP()

        new_points = []

        self.new_lines = []
        self.points_to_remove = []

        # self.current_color_image = color_image

        if USE_LASER_POINTER:
            blurred = cv2.GaussianBlur(ir_frame, (7, 7), 0)
            thresh = cv2.threshold(blurred, BRIGHTNESS_CUTOFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        else:
            thresh = cv2.threshold(ir_frame, BRIGHTNESS_CUTOFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        if DEBUG_MODE:
            cv2.imshow('tresh', thresh)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            new_points = self.detect_ir_points(largest_contour, ir_frame)

        if USE_LASER_POINTER:

            if DEBUG_MODE:
                filtered_contours = np.zeros(thresh.shape)

            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(math.ceil(radius))

                if MIN_PEN_SPOT_RADIUS <= radius <= MAX_PEN_SPOT_RADIUS:
                    if DEBUG_MODE:
                        cv2.drawContours(filtered_contours, [contour], -1, (255), -1)
                        # print('Radius: OK ({})'.format(radius))
                    radius = 2 * MAX_PEN_SPOT_RADIUS
                    # region = color_image[center[1] - radius:center[1] + radius, center[0] - radius: center[0] + radius]
                    # region_ir = ir_frame[center[1] - radius:center[1] + radius, center[0] - radius: center[0] + radius]
                    # if self.__is_red_laser_pointer_spot(region, region_ir):
                    new_points.append(Point(center[0], center[1]))

                    # if DEBUG_MODE:
                        # cv2.circle(ir_frame, center, radius, (0, 0, 0), 2)
                        # cv2.circle(color_image, center, radius, (0, 0, 255), 1)
                else:
                    if DEBUG_MODE:
                        print('Radius: NOT OK ({})'.format(radius))

            if DEBUG_MODE:
                cv2.imshow('filtered', filtered_contours)

        self.active_points = self.merge_points(new_points, change_color)

        if DEBUG_MODE:
            cv2.imshow('spots', ir_frame)

        return self.active_points, self.stored_lines, self.new_lines, self.points_to_remove, self.selected_color

    point_buffer = []

    def detect_ir_points(self, contour, ir_frame, collect_training_data=False):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))

        center_rect, (width, height), angle_rect = cv2.minAreaRect(contour)

        (minVal, brightest_spot, minLoc, maxLoc) = cv2.minMaxLoc(ir_frame)

        # if DEBUG_MODE:
            # print('----------')
            # print('Radius:', radius)
            # print('Radius as float:', radius)
            # radius = round(radius, 2)
            # print('Contour_Area:', contour_area)
            # print('Perimeter:', round(perimeter, 2))
            # contour_area = cv2.contourArea(contour)
            # perimeter = cv2.arcLength(contour, True)
            # print('Brightest_spot:', brightest_spot)

        if width > 0 and height > 0:
            aspect_ratio = round(abs(width/height), 2)
        else:
            aspect_ratio = 0

        if collect_training_data:
            return [radius, brightest_spot, aspect_ratio, x, y]
        else:
            # prediction = self.svm.predict([radius, brightest_spot, aspect_ratio])
            prediction = self.smp.predict(x, y, radius, brightest_spot, aspect_ratio)
            #prediction = self.svm.predict([radius, brightest_spot, x, y])
            if prediction == 0:
                print('Status: Touch')
                new_point = Point(center[0], center[1])
                is_rubbing_event = self.check_rubbing_event(center)
                if is_rubbing_event:
                    new_point.state = State.RUBBING
                return [new_point]
            elif prediction == 1:
                print('Status: Hover')
            elif prediction == 2:
                print('Status: Away')
            else:
                print('Unknown state')

            return []

            '''
            PREVIEW_REGION_SIZE = 20
            # (minVal, brightest_spot, minLoc, maxLoc) = cv2.minMaxLoc(region_ir)
            # if 0.5 < aspect_ratio < 1.5:
            #     print('Aspect_ratio:', aspect_ratio, '-> Looks like a circle')
            # else:
            #     print('Aspect_ratio:', aspect_ratio, '-> Definitely not a circle')
            
            region_ir = ir_frame[center[1] - PREVIEW_REGION_SIZE:center[1] + PREVIEW_REGION_SIZE,
                                 center[0] - PREVIEW_REGION_SIZE: center[0] + PREVIEW_REGION_SIZE]


            brightest_spot_x, brightest_spot_y = maxLoc
            # print('Brightness falloff:',
            #       region_ir[(brightest_spot_y - 2, brightest_spot_x)],
            #       region_ir[(brightest_spot_y - 1, brightest_spot_x)],
            #       region_ir[(brightest_spot_y, brightest_spot_x)],
            #       region_ir[(brightest_spot_y + 1, brightest_spot_x)],
            #       region_ir[(brightest_spot_y + 2, brightest_spot_x)])
            
            MIN_RADIUS_TOUCH = 4.0
            MAX_RADIUS_TOUCH = 5.5

            MIN_BRIGHTNESS_POINT_TOWARDS_CAMERA = 255
            MIN_BRIGHTNESS_TOUCH = 210

            # status = 'Hover > 2 cm'
            status = 'Hover'

            if radius > MAX_RADIUS_TOUCH:
                # Eher dunkel
                if brightest_spot < MIN_BRIGHTNESS_TOUCH:
                    # status = 'Hover between 1 and 1.5 cm'
                    status = 'Hover'

                # Super hell
                #elif brightest_spot >= MIN_BRIGHTNESS_POINT_TOWARDS_CAMERA:
                #     status ='Pointing pen towards camera'

                # Dazwischen
                else:
                    # status = 'Hover between 0.5 and 1 cm'
                    status = 'Hover'
            else:
                if MIN_RADIUS_TOUCH <= radius <= MAX_RADIUS_TOUCH:
                    if brightest_spot > MIN_BRIGHTNESS_TOUCH:
                        status = 'Touch'
                        all_checks_passed = True
                if brightest_spot < 150:
                    status = 'Hover'
                    # status = 'Hover between 1.5 and 2 cm'

            print(Fore.GREEN + 'Status:', status)
            print(Style.RESET_ALL)

            if DEBUG_MODE:
                try:
                    cv2.imshow('POINT', cv2.resize(region_ir, (400, 400)))
                except Exception as e:
                    print(region_ir.shape)
                    print(e)

            # if status == 'Hover' or status == 'Touch':
            if status == 'Touch':
                new_point = Point(center[0], center[1])

                # if status == 'Hover':
                #     new_point.state = State.HOVER

                # if status == 'Touch':
                #     is_rubbing_event = False  # self.check_rubbing_event(center)
                #     if is_rubbing_event:
                #         new_point.state = State.RUBBING

                return [new_point]
            else:
                return []
            '''

    def check_rubbing_event(self, center):
        BUFSIZE = 50
        self.point_buffer.append((center[0], center[1]))
        self.point_buffer = self.point_buffer[-BUFSIZE:]
        # calculate bb
        xs = [p[0] for p in self.point_buffer]
        ys = [p[1] for p in self.point_buffer]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        xdelta = xmax - xmin
        ydelta = ymax - ymin
        if len(self.point_buffer) > 1:
            xsd = statistics.stdev(xs)
            ysd = statistics.stdev(ys)
        else:
            xsd, ysd = 0, 0
        print(xdelta, ydelta, xsd, ysd, len(self.point_buffer))
        if xdelta > 8 or ydelta > 8:
            self.point_buffer = []
        if len(self.point_buffer) > 30 and (xsd + ysd) > 3 and (xdelta + ydelta) > 6:
            print('\a')
            self.point_buffer = []
            return True
        else:
            return False

        ## dead code ahead


        signs = []
        pointy = 0
        for pp in zip(self.point_buffer, self.point_buffer[1:]):
            dx = pp[1][0] - pp[0][0]
            sx = -1 if dx < 0 else 1
            dy = pp[1][1] - pp[0][1]
            sy = -1 if dy < 0 else 1
            if abs(dx) > 2 or abs(dy) > 2:
                signs.append((sx, sy))
        for sp in zip(signs, signs[1:]):
            if sp[0][0] != sp[1][0] or sp[0][1] != sp[1][1]:
                pointy += 1
        pointy += 1
        print(f"> {pointy}")
        if pointy > 2 and xdelta < 20 and ydelta < 20:
            print('\a')
            #print('\a')
            #print('\a')
            print('Rubbing')
            self.point_buffer = []
            return True
        return False

    # Implementation inspired by the approach described in the paper
    # "DIRECT: Making Touch Tracking on Ordinary Surfaces Practical with Hybrid Depth-Infrared Sensing."
    # by Xiao, R., Hudson, S., & Harrison, C. (2016).
    # See: https://github.com/nneonneo/direct-handtracking/blob/c199dd61ab097f3b3f155798c2519828352d9bdb/ofx/apps
    # /handTracking/direct/src/IRDepthTouchTracker.cpp
    # Check for each new point if it has been present in previous frames and assign unique IDs
    def merge_points(self, new_points, change_color):

        final_points = []
        now = round(time.time() * 1000)

        # for point in self.active_points:
        #     print(point)

        # Iterate over copy of list
        # If a point has been declared a "Click Event" in the last frame, this event is now over and we can delete it.
        for active_point in self.active_points[:]:
            if active_point.state == State.CLICK:
                xs = []
                ys = []
                for x, y in active_point.history:
                    xs.append(x)
                    ys.append(y)
                dx = abs(max(xs) - min(xs))
                dy = abs(max(ys) - min(ys))
                if dx < 5 and dy < 5:
                    print('CLICK')
                    # print('\a')

                    # # We have a new click event. Check if it belongs to a previous click event (-> Double click)
                    # active_point = self.check_if_double_click(now, active_point, change_color)
                    #
                    # if active_point.state == State.DOUBLE_CLICK:
                    #     # Pass this point forward to the final return call because we want to send at least one alive
                    #     # message for the double click event
                    #     final_points.append(active_point)
                    #
                    #     # Give the Double Click event a different ID from the previous click event
                    #     # active_point.id = self.highest_id
                    #     # self.highest_id += 1
                    # else:
                    #     # We now know that the current click event is no double click event,
                    #     # but it might be the first click of a future double click. So we remember it.
                    #     self.double_click_candidates.append(active_point)

                self.points_to_remove.append(active_point)
                self.active_points.remove(active_point)
            elif active_point.state == State.RUBBING:
                print('REMOVE RUBBING EVENT')

                # Replace Rubbing event with double click
                active_point.state = State.DOUBLE_CLICK

                self.points_to_remove.append(active_point)
                self.active_points.remove(active_point)

        distances = []

        for i in range(len(self.active_points)):
            for j in range(len(new_points)):
                distance_between_points = distance.euclidean(self.active_points[i].get_coordinates(),
                                                             new_points[j].get_coordinates())

                # If distance is large enough, there is no need to check if the touch point already exists
                if distance_between_points > DISTANCE_MOVEMENT_BETWEEN_FRAMES_THRESHOLD:
                    print('DISTANCE TOO LARGE')
                    continue
                distances.append([i, j, distance_between_points])

        # Sort list of lists by third element, in this case the distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        distances.sort(key=lambda x: x[2])

        for entry in distances:
            active_touch_point = self.active_points[entry[0]]
            new_touch_point = new_points[entry[1]]

            # id = -1 means that it is a new point
            if active_touch_point.id == -1 or new_touch_point.id != -1:
                continue

            # Move ID and other important information from the active touch point into the new touch point
            new_touch_point.id = active_touch_point.id
            new_touch_point.first_appearance = active_touch_point.first_appearance

            if new_touch_point.state != active_touch_point.state:
                # if active_touch_point.state == State.RUBBING:
                #     print('FOUND ACTIVE POINT WITH STATE RUBBING')

                if new_touch_point.state == State.RUBBING:
                    print('Found new RUBBING EVENT')
                    # Do not overwrite if rubbing event
                elif active_touch_point.state == State.HOVER and new_touch_point.state != State.HOVER:
                    print('HOVER EVENT to touch event')
                else: #  new_touch_point.state != State.RUBBING:
                    new_touch_point.state = active_touch_point.state

            new_touch_point.history = active_touch_point.history

            # TODO: Check if this ID reset is needed
            active_touch_point.id = -1

            new_touch_point.x = int(SMOOTHING_FACTOR * (new_touch_point.x - active_touch_point.x) + active_touch_point.x)
            new_touch_point.y = int(SMOOTHING_FACTOR * (new_touch_point.y - active_touch_point.y) + active_touch_point.y)

            # new_touch_point.x = active_point.x
            # new_touch_point.y = active_point.y

        for new_point in new_points:
            new_point.missing = False
            new_point.last_seen_timestamp = now

        for active_point in self.active_points:

            time_since_last_seen = now - active_point.last_seen_timestamp

            # If it is not a new point (ID != -1) and the point is not yet declared as missing:
            if active_point.id != -1 and (not active_point.missing or time_since_last_seen < TIME_POINT_MISSING_THRESHOLD_MS):
                if not active_point.missing:
                    active_point.last_seen_timestamp = now

                active_point.missing = True
                new_points.append(active_point)

            elif active_point.id != -1:
                if active_point.state == State.NEW:
                    if now - active_point.first_appearance < MIN_TIME_EXISTING_MS:
                        print('TOO LITTLE TIME CLICK')
                    active_point.state = State.CLICK
                    new_points.append(active_point)
                elif active_point.state == State.DRAG:
                    # End of a drag event
                    print('DRAG END')
                    if now - active_point.first_appearance < MIN_TIME_EXISTING_MS:
                        print('TOO LITTLE TIME DRAG')
                    self.points_to_remove.append(active_point)
                    self.stored_lines.append(np.array(active_point.history))
                    self.new_lines.append(active_point.history)
                elif active_point.state == State.HOVER:
                    # End of a Hover event
                    print('HOVER EVENT END')
                    self.points_to_remove.append(active_point)

        for new_point in new_points:
            if new_point.id == -1:
                new_point.id = self.highest_id
                self.highest_id += 1

            final_points.append(new_point)

        for point in final_points:
            point.history.append((point.x, point.y))
            time_since_first_appearance = now - point.first_appearance
            if point.state != State.CLICK and point.state != State.DOUBLE_CLICK and time_since_first_appearance > CLICK_THRESH_MS:
                if point.state == State.NEW:
                    # Start of a drag event
                    print('DRAG START')
                    point.state = State.DRAG
                elif point.state == State.RUBBING:
                    print('DETECTED RUBBING EVENT!')
                elif point.state == State.HOVER:
                    print('DETECTED Hover EVENT!')
                # else:
                #     point.state = State.DRAG

        if len(final_points) > MAX_NUM_POINTS:
            print('TOO MANY POINTS!')
            final_points = final_points[0:MAX_NUM_POINTS]

        final_points = self.remove_accidental_point_clusters(final_points)

        return final_points

    def remove_accidental_point_clusters(self, final_points):
        len_before = len(final_points)

        ids_to_remove = []
        for i in range(len(final_points)):
            current_point = final_points[i]
            j = i
            while j < len(final_points) - 1:
                next_point = final_points[j + 1]
                j += 1
                distance_between_points = distance.euclidean(current_point.get_coordinates(),
                                                             next_point.get_coordinates())
                print('comparing {} and {}'.format(current_point.id, next_point.id))
                if distance_between_points < MIN_DISTANCE_BETWEEN_POINTS:
                    print('POINTS TOO CLOSE!')
                    ids_to_remove.append(current_point.id)
                    ids_to_remove.append(next_point.id)

        for id in ids_to_remove:
            for point in final_points[:]:
                if point.id == id:
                    final_points.remove(point)

        if len_before > 1:
            print('Before: {}; After: {}'.format(len_before, len(final_points)))
        return final_points

    def check_if_double_click(self, now, click_event_point, change_color):
        if len(self.double_click_candidates) > 0:

            # iterate over copy of list
            for previous_click_event in self.double_click_candidates[:]:
                if now - previous_click_event.last_seen_timestamp > 2 * CLICK_THRESH_MS:
                    self.double_click_candidates.remove(previous_click_event)

            # Check all remaining click events:
            for previous_click_event in self.double_click_candidates:
                if True:  # TODO: Here we need to check if pos of point close to previous click event
                    print('DOUBLE CLICK')
                    # print('\a')
                    # print('\a')
                    click_event_point.state = State.DOUBLE_CLICK

                    if change_color:
                        b, g, r = self.current_color_image[click_event_point.y, click_event_point.x]
                        self.selected_color = [b, g, r]

                    self.double_click_candidates.remove(previous_click_event)
                    break

        return click_event_point

    # def delete_stored_lines(self):
    #     self.stored_lines = []
    #     self.new_lines = []
    #     print(len(self.new_lines))

    # def __is_red_laser_pointer_spot(self, region, region_ir):
    #     if DONT_CHECK_FOR_COLOR:
    #         return True
    #     try:
    #         # region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    #
    #         # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    #         # mask0 = cv2.inRange(region_hsv, LOWER_RED_BOTTOM_RANGE, UPPER_RED_BOTTOM_RANGE)
    #         # mask1 = cv2.inRange(region_hsv, LOWER_RED_TOP_RANGE, UPPER_RED_TOP_RANGE)
    #         # mask = mask0 + mask1  # combine masks
    #
    #         # mask = cv2.inRange(region_hsv, LOWER_SPOT, UPPER_SPOT)
    #
    #         # cv2.imshow('region', region_ir)
    #
    #         detected_circles = cv2.HoughCircles(region_ir,
    #                                             cv2.HOUGH_GRADIENT, 1, 5, param1=1,
    #                                             param2=5, minRadius=MIN_PEN_SPOT_RADIUS, maxRadius=2*MAX_PEN_SPOT_RADIUS)
    #
    #         passes_circle_check = False
    #         if detected_circles is not None:
    #             if len(detected_circles) == 1:
    #                 passes_circle_check = True
    #
    #         if DEBUG_MODE:
    #             region_large = cv2.resize(region, (400, 400), interpolation=cv2.INTER_AREA)
    #             region_large = cv2.putText(
    #                 img=region_large,
    #                 text="Is Circle: {}".format(passes_circle_check),
    #                 org=(10, 10),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX,
    #                 fontScale=1,
    #                 color=(0, 0, 255),
    #                 thickness=1
    #             )
    #             # cv2.imshow('region large', region_large)
    #             # cv2.imshow('region IR large', cv2.resize(region_ir, (400, 400), interpolation=cv2.INTER_AREA))
    #             # cv2.imshow('red mask', cv2.resize(mask, (400, 400), interpolation=cv2.INTER_AREA))
    #         # red_percentage = int((cv2.countNonZero(mask) / mask.size) * 100)
    #
    #         # if red_percentage >= MIN_PERCENTAGE_RED:
    #             # if DEBUG_MODE:
    #             #     print('COLOR CHECK: OK ({})'.format(str(red_percentage) + '%'))
    #             return True
    #         # if DEBUG_MODE:
    #         #     print('COLOR CHECK: NOT OK ({})'.format(str(red_percentage) + '%'))
    #     except Exception as e:
    #         print(e)
    #     return False