import os
import random
import sys
import threading
from enum import Enum
import time
import numpy as np
import skimage
from tensorflow import keras
import socket

import flir_blackfly_s
from tflite import LiteModel
import datetime
from scipy.spatial import distance
from skimage.feature import peak_local_max

from logitech_brio import LogitechBrio
from AnalogueDigitalDocumentsDemo import AnalogueDigitalDocumentsDemo

from cv2 import cv2

MODEL_PATH = 'cnn'  # Put the folder path here for the desired cnn
CROP_IMAGE_SIZE = 48  # Currently 48x48 Pixel

# Simple Smoothing of the output points. 1 -> No smoothing; 0.5 -> Calculate a new point and use 50% of the previous
# points and 50% of the new points location.
SMOOTHING_FACTOR = 0.2  # Value between 0 and 1, depending on if the old or the new value should count more.

# Amount of time a pen can be missing until the pen event will be ended
TIME_POINT_MISSING_THRESHOLD_MS = 15

# Hover will be selected over Draw if Hover Event is within the last X event states.
# Enable this if too many unwanted short lines appear while drawing
HOVER_WINS = False
NUM_HOVER_EVENTS_TO_END_LINE = 6  # We expect at least X Hover Events in a row to end that pen event
NUM_CHECK_LAST_EVENT_STATES = 3  # Check the last X Pen Events if they contain a hover event

MIN_BRIGHTNESS_FOR_PREDICTION = 50  # A spot in the camera image needs to have at least X brightness to be considered.

MAX_DISTANCE_FOR_MERGE = 500  # Maximum Distance between two points for them to be able to merge

USE_MAX_DISTANCE_DRAW = False
MAX_DISTANCE_DRAW = 500  # Maximum allowed distance in pixel between two points in order to be considered for the same line ID

DEBUG_MODE = False  # Enable for Debug print statements and preview windows
SEND_TO_FRONTEND = True  # Enable if points should be forwarded to the sdl frontend
ENABLE_FIFO_PIPE = False
ENABLE_UNIX_SOCKET = True
UNIX_SOCK_NAME = 'uds_test'

LATENCY_MEASURING_MODE = False

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

CAMERA_WIDTH = 1920  # 848
CAMERA_HEIGHT = 1200  # 480

# Allowed states for CNN prediction
STATES = ['draw', 'hover', 'hover_far', 'undefined']

# Enable to collect training images if you want to retrain the CNN (see train_network.ipynb)
TRAINING_DATA_COLLECTION_MODE = False
ACTIVE_LEARNING_COLLECTION_MODE = False
TRAIN_STATE = 'hover_far_1_{}_{}'.format(flir_blackfly_s.EXPOSURE_TIME_MICROSECONDS, flir_blackfly_s.GAIN)
TRAIN_PATH = 'training_images/2022-08-19'
TRAIN_IMAGE_COUNT = 3000


# Decorator to print the run time of a single function
# Based on: https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            return_value = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return return_value

        return wrapper

    return timeit_decorator


# Enum to define State of a Point
class State(Enum):
    NEW = -1  # A new event where it is not yet sure what state it will have
    CLICK = 0
    DRAG = 1
    DOUBLE_CLICK = 2
    HOVER = 3


# Representation of a single pen event
class PenEvent:

    def __init__(self, x, y, state=State.NEW):
        self.x = x
        self.y = y
        self.id = -1

        self.missing = False
        self.last_seen_timestamp = 0

        self.first_appearance = round(time.time() * 1000)
        self.state = state
        self.history = []  # All logged x and y positions as tuples
        self.state_history = [state]  # All logged states (Hover, Draw, ...)

    def get_coordinates(self):
        return tuple([self.x, self.y])

    def __repr__(self):
        return 'Pen Event {} at ({}, {}). Type: {}. Num Points: {}'.format(str(self.id), str(self.x), str(self.y),
                                                                           self.state, len(self.history))


class IRPen:
    active_pen_events = []
    new_pen_events = []

    highest_id = 0  # Assign each new pen event a new id. This variable keeps track of the highest number.

    # If a pen event is over, it will be deleted. But its points will be stored here to keep track of all drawn lines.
    stored_lines = []
    pen_events_to_remove = []  # Points that got deleted from active_points in the current frame

    double_click_candidates = []

    # Only needed during training TRAINING_DATA_COLLECTION_MODE. Keeps track of the number of saved images
    saved_image_counter = 0
    num_saved_images_cam_0 = 0
    num_saved_images_cam_1 = 0

    active_learning_counter = 0
    active_learning_state = 'hover'

    last_coords = (0, 0)
    last_frame_time = time.time()

    def __init__(self):
        # Init Keras
        keras.backend.clear_session()
        self.keras_lite_model = LiteModel.from_keras_model(keras.models.load_model(MODEL_PATH))
        # self.keras_lite_model = keras.models.load_model(MODEL_PATH)

        if TRAINING_DATA_COLLECTION_MODE or ACTIVE_LEARNING_COLLECTION_MODE:
            if not os.path.exists(os.path.join(TRAIN_PATH, TRAIN_STATE)):
                os.makedirs(os.path.join(TRAIN_PATH, TRAIN_STATE))
            else:
                print('WARNING: FOLDER ALREADY EXISTS. PLEASE EXIT TRAINING MODE IF THIS WAS AN ACCIDENT')
                time.sleep(100000)

    def save_training_image(self, img, pos, camera_id):
        num_wait_frames = 3
        if self.saved_image_counter == 0:
            print('Starting in 5 Seconds')
            time.sleep(5)

        self.saved_image_counter += 1
        if self.saved_image_counter % num_wait_frames == 0:

            cv2.imwrite(
                f'{TRAIN_PATH}/{TRAIN_STATE}/{TRAIN_STATE}_{int(self.saved_image_counter / num_wait_frames)}_{pos[0]}_{pos[1]}.png',
                img)
            print(
                f'saving frame {int(self.saved_image_counter / num_wait_frames)}/{TRAIN_IMAGE_COUNT} from camera {camera_id}')
            if camera_id == 0:
                self.num_saved_images_cam_0 += 1
            elif camera_id == 1:
                self.num_saved_images_cam_1 += 1

        if self.saved_image_counter / num_wait_frames >= TRAIN_IMAGE_COUNT:
            print('FINISHED COLLECTING TRAINING IMAGES. Saved {} images from cam 0 and {} images from cam 1'.format(
                self.num_saved_images_cam_0, self.num_saved_images_cam_1))
            time.sleep(10)
            sys.exit(0)

    def transform_coords_to_output_res(self, x, y, transform_matrix):
        try:
            coords = np.array([x, y, 1])

            transformed_coords = transform_matrix.dot(coords)
            # Normalize coordinates by dividing by z
            # transformed_coords = (int(transformed_coords[0] / transformed_coords[2]),
            #                       int(transformed_coords[1] / transformed_coords[2]))

            transformed_coords = (
            transformed_coords[0] / transformed_coords[2], transformed_coords[1] / transformed_coords[2])

            return transformed_coords
        except Exception as e:
            print(e)
            print('Error in transform_coords_to_output_res(). Maybe the transform_matrix is malformed?')
            print('This error could also appear if CALIBRATION_MODE is still enabled in flir_blackfly_s.py')
            time.sleep(5)
            sys.exit(1)

    # @timeit('Pen Events')
    def get_ir_pen_events_multicam(self, camera_frames, transform_matrices):
        new_pen_events = []

        self.new_pen_events = []
        self.pen_events_to_remove = []

        brightness_values = []

        predictions = []
        rois = []
        roi_coords = []
        subpixel_coords = []

        debug_distances = []

        # # WIP new approach for multiple pens
        # WIP_APPROACH_MULTI_PEN = False
        # if WIP_APPROACH_MULTI_PEN:
        #     for i, frame in enumerate(camera_frames):
        #
        #         # Add a new sublist for each frame
        #         predictions.append([])
        #         rois.append([])
        #         roi_coords.append([])
        #         brightness_values.append([])
        #         subpixel_coords.append([])
        #
        #         rois_new, roi_coords_new, max_brightness_values = self.get_all_rois(frame)
        #
        #         for j, pen_event_roi in enumerate(rois_new):
        #
        #             if TRAINING_DATA_COLLECTION_MODE:
        #                 self.save_training_image(pen_event_roi, (roi_coords_new[j][0], roi_coords_new[j][1]))
        #                 continue
        #
        #             prediction, confidence = self.predict(pen_event_roi)
        #
        #             predictions[i].append(prediction)
        #             rois[i].append(pen_event_roi)
        #
        #             transformed_coords = self.transform_coords_to_output_res(roi_coords_new[j][0], roi_coords_new[j][1], transform_matrices[i])
        #             roi_coords[i].append(transformed_coords)
        #
        #             brightness_values[i].append(max_brightness_values[j])
        #
        #             (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)
        #             subpixel_coords[i].append((x, y))
        #
        #     new_pen_events = self.generate_new_pen_events(subpixel_coords, predictions, brightness_values)
        #
        #     # This function needs to be called even if there are no new pen events to update all existing events
        #     self.active_pen_events = self.merge_pen_events(new_pen_events)
        #
        #     return self.active_pen_events, self.stored_lines, self.new_pen_events, self.pen_events_to_remove, debug_distances

        for i, frame in enumerate(camera_frames):
            # TODO: Get here all spots and not just one

            pen_event_roi, brightest, (x, y) = self.crop_image(frame)

            if brightest > MIN_BRIGHTNESS_FOR_PREDICTION and pen_event_roi.shape[0] == CROP_IMAGE_SIZE and \
                    pen_event_roi.shape[1] == CROP_IMAGE_SIZE:
                if TRAINING_DATA_COLLECTION_MODE:
                    self.save_training_image(pen_event_roi, (x, y), i)
                    continue

                rois.append(pen_event_roi)
                transformed_coords = self.transform_coords_to_output_res(x, y, transform_matrices[i])
                roi_coords.append(transformed_coords)

                brightness_values.append(np.sum(pen_event_roi))

                (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)

                subpixel_coords.append((x, y))
                debug_distances.append((x, y))

        # If we see only one point:
        if len(subpixel_coords) == 1:
            prediction, confidence = self.predict(rois[0])
            # print('One Point', prediction, confidence)
            if prediction == 'draw':
                # print('One point draw')
                new_ir_pen_event = PenEvent(subpixel_coords[0][0], subpixel_coords[0][1], State.DRAG)
                new_pen_events.append(new_ir_pen_event)

            elif prediction == 'hover':
                # print('One point hover')
                new_ir_pen_event = PenEvent(subpixel_coords[0][0], subpixel_coords[0][1], State.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                print('Error: Unknown state')

        # If we see two points
        elif len(subpixel_coords) == 2:
            # print('Two points')
            (center_x, center_y) = self.get_center(subpixel_coords[0], subpixel_coords[1])
            debug_distances.append((center_x, center_y))

            distance_between_points = distance.euclidean(subpixel_coords[0],
                                                         subpixel_coords[1])

            # print('DISTANCE:', distance_between_points, flush=True)

            if USE_MAX_DISTANCE_DRAW and distance_between_points > MAX_DISTANCE_DRAW:
                print('Distance too large -> Hover', distance_between_points)
                # Calculate center between the two points

                final_prediction = 'hover'
            else:
                for pen_event_roi in rois:
                    prediction, confidence = self.predict(pen_event_roi)
                    # print('Two Points', prediction, confidence)
                    predictions.append(prediction)

                if all(x == predictions[0] for x in predictions):
                    # The predictions for all cameras are the same
                    final_prediction = predictions[0]
                    # print('Agreement on prediction')
                else:
                    brightest_image_index = brightness_values.index(max(brightness_values))
                    # print('Brightness vs: {} > {}'.format(max(brightness_values), brightness_values))
                    # print('Disagree -> roi {} wins because it is brighter'.format(brightest_image_index))
                    # There is a disagreement
                    # Currently we then use the prediction of the brightest point in all camera frames
                    final_prediction = predictions[brightest_image_index]
                    # TODO: OR HOVER WINS HERE

            if final_prediction == 'draw':
                # print('Status: Touch')
                new_ir_pen_event = PenEvent(center_x, center_y, State.DRAG)
                new_pen_events.append(new_ir_pen_event)

            elif final_prediction == 'hover':
                # print('Status: Hover')
                new_ir_pen_event = PenEvent(center_x, center_y, State.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                print('Error: Unknown state')

        if LATENCY_MEASURING_MODE:
            if len(subpixel_coords) == 0:
                new_ir_pen_event = PenEvent(0, 0, State.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                new_pen_events[-1] = PenEvent(center_x, center_y, State.DRAG)

        # For distance/velocity calculation
        # current_frame_time = time.time()
        # delta_time = current_frame_time - self.last_frame_time
        # dist = distance.euclidean((center_x, center_y),
        #                           self.last_coords)
        # self.last_coords = (center_x, center_y)
        # self.last_frame_time = current_frame_time
        # print('LOG {}, {}'.format(distance_between_points, abs(dist) / delta_time), flush=True)

        # This function needs to be called even if there are no new pen events to update all existing events
        self.active_pen_events = self.merge_pen_events_single(new_pen_events)
        self.rois = rois

        return self.active_pen_events, self.stored_lines, self.new_pen_events, self.pen_events_to_remove, debug_distances, rois

    def generate_new_pen_events(self, subpixel_coords, predictions, brightness_values):

        new_pen_events = []

        zeros = np.zeros((2160, 3840, 3), 'uint8')
        for i, point_left_cam in enumerate(subpixel_coords[0]):
            if predictions[0][i] == 'draw':
                cv2.circle(zeros, point_left_cam, 20, (255, 0, 0), 1)
            else:
                cv2.circle(zeros, point_left_cam, 20, (255, 0, 0), -1)

        for i, point_right_cam in enumerate(subpixel_coords[1]):
            if predictions[1][i] == 'draw':
                cv2.circle(zeros, point_right_cam, 20, (0, 0, 255), 1)
            else:
                cv2.circle(zeros, point_right_cam, 20, (0, 0, 255), -1)

        # Deal with single points first
        if len(subpixel_coords[0]) == 0 and len(subpixel_coords[1]) > 0 or len(subpixel_coords[0]) > 0 and len(
                subpixel_coords[1]) == 0:
            for i, point in enumerate(subpixel_coords[0]):
                print('Single point at', point)
                new_pen_events.append(self.generate_new_pen_event(predictions[0][i], point[0], point[1]))
            for i, point in enumerate(subpixel_coords[1]):
                print('Single point at', point)
                new_pen_events.append(self.generate_new_pen_event(predictions[1][i], point[0], point[1]))

        elif len(subpixel_coords[0]) > 0 and len(subpixel_coords[1]) > 0:
            shortest_distance_point_pairs = self.calculate_distances_between_all_points(subpixel_coords[0],
                                                                                        subpixel_coords[1],
                                                                                        as_objects=False)

            used_left = []
            used_right = []
            for entry in shortest_distance_point_pairs:
                if entry[0] not in used_left and entry[1] not in used_right:
                    y1 = subpixel_coords[0][entry[0]][1]
                    y2 = subpixel_coords[1][entry[1]][1]
                    if abs(y1 - y2) < MAX_DISTANCE_DRAW:
                        if predictions[0][entry[0]] == 'draw' or predictions[1][entry[1]] == 'draw':
                            dist = entry[2]
                            if dist > 50:
                                continue

                        (center_x, center_y) = self.get_center(subpixel_coords[0][entry[0]],
                                                               subpixel_coords[1][entry[1]])
                        cv2.line(zeros, subpixel_coords[0][entry[0]], subpixel_coords[1][entry[1]], (255, 255, 255), 3)
                        used_left.append(entry[0])
                        used_right.append(entry[1])

                        # TODO: Compare brightness and use the prediction of the brighter point
                        prediction_a = predictions[0][entry[0]]
                        prediction_b = predictions[1][entry[1]]

                        new_pen_events.append(self.generate_new_pen_event(prediction_a, center_x, center_y))

            for i in range(len(subpixel_coords[0])):
                if i not in used_left:
                    print('Point {} on cam 0 remains'.format(i))
                    new_pen_events.append(self.generate_new_pen_event(predictions[0][i], subpixel_coords[0][i][0],
                                                                      subpixel_coords[0][i][1]))

            for i in range(len(subpixel_coords[1])):
                if i not in used_right:
                    print('Point {} on cam 1 remains'.format(i))
                    new_pen_events.append(self.generate_new_pen_event(predictions[1][i], subpixel_coords[1][i][0],
                                                                      subpixel_coords[1][i][1]))

            # print(shortest_distance_point_pairs)
        # print(new_pen_events)

        # for point_left_cam in subpixel_coords[0]:
        #     for point_right_cam in subpixel_coords[1]:
        #         dist_between_y = int(abs(point_left_cam[1] - point_right_cam[1]))
        #         distance_between_points = int(distance.euclidean(point_left_cam, point_right_cam))
        #
        #         print(distance_between_points, dist_between_y)

        # cv2.imshow('preview', zeros)

        return new_pen_events

    def generate_new_pen_event(self, prediction, x, y):
        if prediction == 'draw':
            # print('Status: Touch')
            new_ir_pen_event = PenEvent(x, y, State.DRAG)
            # new_ir_pen_event.state = State.DRAG
            return new_ir_pen_event

        elif prediction == 'hover':
            # print('Status: Hover')
            new_ir_pen_event = PenEvent(x, y, State.HOVER)
            # new_ir_pen_event.state = State.HOVER
            return new_ir_pen_event
        else:
            print('Error: Unknown state')
            sys.exit(1)

    # Calculate the center point between two given points
    def get_center(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2

        x_dist = abs(x1 - x2) / 2
        y_dist = abs(y1 - y2) / 2
        return min(x1, x2) + x_dist, min(y1, y2) + y_dist

    def get_all_rois(self, img, size=CROP_IMAGE_SIZE):

        rois = []
        roi_coords = []
        max_brightness_values = []

        for i in range(10):
            margin = int(size / 2)
            _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)

            # Dead pixel fix
            if max_x == 46 and max_y == 565:
                # TODO: Find solution for dead pixel
                continue

            if brightest < MIN_BRIGHTNESS_FOR_PREDICTION:
                break

            img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin].copy()

            # print('Shape in crop 1:', img_cropped.shape, max_x, max_y)

            if img_cropped.shape == (size, size):
                rois.append(img_cropped)
                roi_coords.append((max_x, max_y))
                max_brightness_values.append(int(brightest))
            else:
                print('TODO: WRONG SHAPE')

            img.setflags(write=True)
            img[max_y - margin: max_y + margin, max_x - margin: max_x + margin] = 0
            img.setflags(write=False)

        return rois, roi_coords, max_brightness_values

    def crop_image(self, img, size=CROP_IMAGE_SIZE):
        if len(img.shape) == 3:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_grey = img
        margin = int(size / 2)
        _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img_grey)

        img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin]

        # print('Shape in crop 1:', img_cropped.shape, max_x, max_y)

        # TODO: Improve this
        if img_cropped.shape[0] != size or img_cropped.shape[1] != size:
            img_cropped, brightest, (max_x, max_y) = self.crop_image_2(img)

        # print('Shape in crop 2:', img_cropped.shape, max_x, max_y)
        # img_cropped_large = cv2.resize(img_cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('large', img_cropped_large)
        return img_cropped, brightest, (max_x, max_y)

    def crop_image_2(self, img, size=CROP_IMAGE_SIZE):
        # print('using crop_image_2() function')
        margin = int(size / 2)
        brightest = int(np.max(img))
        _, thresh = cv2.threshold(img, brightest - 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print(contours)

        contours = contours[0] if len(contours) == 2 else contours[1]

        x_values = []
        y_values = []
        for cnt in contours:
            for point in cnt:
                point = point[0]
                # print(point)
                x_values.append(point[0])
                y_values.append(point[1])

        # print('x', np.max(x_values), np.min(x_values))
        # print('y', np.max(y_values), np.min(y_values))
        d_x = np.max(x_values) - np.min(x_values)
        d_y = np.max(y_values) - np.min(y_values)
        center_x = int(np.min(x_values) + d_x / 2)
        center_y = int(np.min(y_values) + d_y / 2)
        # print(center_x, center_y)

        left = np.max([0, center_x - margin])
        top = np.max([0, center_y - margin])

        # print(left, top)

        if left + size >= img.shape[1]:
            # left -= (left + size - img.shape[1] - 1)
            left = img.shape[1] - size - 1
        if top + size >= img.shape[0]:
            # top -= (top + size - img.shape[0] - 1)
            top = img.shape[0] - size - 1

        # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)
        img_cropped = img[top: top + size, left: left + size]
        return img_cropped, np.max(img_cropped), (left + margin, top + margin)

    # @timeit('Predict')
    def predict(self, img):
        # if len(img.shape) == 3:
        #     print(img[10,10,:])
        #     img = img[:, :, :2]
        #     print(img[10, 10, :], 'after')
        # img = img.astype('float32') / 255
        # if len(img.shape) == 3:
        #     img = img.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 2)
        # else:
        #     img = img.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 1)

        img_reshaped = img.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 1)
        prediction = self.keras_lite_model.predict(img_reshaped)
        if not prediction.any():
            return STATES[-1], 0
        state = STATES[np.argmax(prediction)]
        confidence = np.max(prediction)

        if ACTIVE_LEARNING_COLLECTION_MODE:
            if state != self.active_learning_state:
                cv2.imwrite(f'{TRAIN_PATH}/{TRAIN_STATE}/{TRAIN_STATE}_{self.active_learning_counter}.png', img)
                print(f'saving frame {self.active_learning_counter}')
                self.active_learning_counter += 1

        # print(state)
        if state == 'hover_far':
            state = 'hover'
        return state, confidence

    # TODO: Fix possible offset
    def find_pen_position_subpixel_crop(self, roi, center_original):
        w = roi.shape[0]
        h = roi.shape[1]
        # print('1', ir_image.shape)
        # center_original = (coords_original[0] + w/2, coords_original[1] + h/2)

        factor_w = WINDOW_WIDTH / CAMERA_WIDTH
        factor_h = WINDOW_HEIGHT / CAMERA_HEIGHT

        new_w = int(w * factor_w)
        new_h = int(h * factor_h)
        top_left_scaled = (center_original[0] * factor_w - new_w / 2, center_original[1] * factor_h - new_h / 2)

        # print(w, h, factor_w, factor_h, new_w, new_h, center_original, top_left_scaled)

        # cv2.imshow('roi', roi)
        # TODO:
        # print('2', ir_image_grey.shape)
        # Set all pixels
        _, thresh = cv2.threshold(roi, np.max(roi) - 1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh', thresh)

        # TODO: resize only cropped area
        # thresh_large = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        thresh_large = skimage.transform.resize(thresh, (new_w, new_h), mode='edge', anti_aliasing=False,
                                                anti_aliasing_sigma=None, preserve_range=True, order=0)
        # thresh_large_preview = cv2.cvtColor(thresh_large.copy(), cv2.COLOR_GRAY2BGR)

        contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        min_radius = thresh_large.shape[0]
        smallest_contour = contours[0]
        min_center = (0, 0)

        # print(len(contours))
        # Find the smallest contour if there are multiple (we want to find the pen tip, not its light beam
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                smallest_contour = contour
                min_center = (round(x), round(y))

        min_radius = 1 if min_radius < 1 else min_radius

        if len(smallest_contour) < 4:
            cX, cY = min_center
            print('small contour')
        else:
            # Find the center of the contour using OpenCV Moments

            M = cv2.moments(smallest_contour)
            # calculate x,y coordinate of center
            try:
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])

                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]

            except Exception as e:
                print(e)
                print('Error in find_pen_position_subpixel_crop()')
                print('Test', min_radius, min_center, len(smallest_contour))
                # time.sleep(5)
                cX = 0
                cY = 0

        # print(cX, cY)

        # position = (int(top_left_scaled[0] + cX), int(top_left_scaled[1] + cY))
        position = (top_left_scaled[0] + cX, top_left_scaled[1] + cY)

        # thresh_large_preview = cv2.drawContours(thresh_large_preview, [smallest_contour], 0, (0, 0, 255), 1)
        # thresh_large_preview = cv2.circle(thresh_large_preview, min_center, round(min_radius), (0, 255, 0), 1)
        # thresh_large_preview = cv2.circle(thresh_large_preview, min_center, 0, (0, 255, 0), 1)
        # thresh_large_preview = cv2.circle(thresh_large_preview, (round(cX), round(cY)), 0, (0, 128, 128), 1)
        # cv2.imshow('thresh_large', thresh_large_preview)

        # print(min_radius, cX, cY, position)

        return position, min_radius

    # Simplified function when only one pen can be present at the time
    def merge_pen_events_single(self, new_pen_events):
        now = round(time.time() * 1000)  # Get current timestamp

        if len(self.active_pen_events) == 0 and len(new_pen_events) == 1:
            # print('New event')
            pass

        elif len(self.active_pen_events) == 1 and len(new_pen_events) == 0:
            active_pen_event = self.active_pen_events[0]

            # print('Remaining active event but now new event to merge with')
            time_since_last_seen = now - active_pen_event.last_seen_timestamp

            if time_since_last_seen < TIME_POINT_MISSING_THRESHOLD_MS:
                new_pen_events.append(active_pen_event)
            else:
                print(
                    'PEN Event {} ({} points) gets deleted due to inactivity with state {}'.format(active_pen_event.id,
                                                                                                   len(active_pen_event.history),
                                                                                                   active_pen_event.state))
                if len(active_pen_event.history) > 0:
                    self.stored_lines.append(np.array(active_pen_event.history))

        elif len(self.active_pen_events) == 1 and len(new_pen_events) == 1:
            # print('Merge new and old')
            last_pen_event = self.active_pen_events[0]
            new_pen_event = new_pen_events[0]

            distance_between_points = distance.euclidean(last_pen_event.get_coordinates(),
                                                         new_pen_event.get_coordinates())

            if distance_between_points > MAX_DISTANCE_FOR_MERGE:
                print('Distance too large. End active event and start new')
                print('PEN Event {} ({} points) gets deleted because distance to new event is too large'.format(
                    last_pen_event.id,
                    len(last_pen_event.history)))

                if len(last_pen_event.history) > 0:
                    self.stored_lines.append(np.array(last_pen_event.history))
            else:

                new_pen_event.id = last_pen_event.id
                new_pen_event.first_appearance = last_pen_event.first_appearance
                new_pen_event.state_history = last_pen_event.state_history
                new_pen_event.state_history.append(new_pen_event.state)
                new_pen_event.last_seen_timestamp = now

                if HOVER_WINS:  # Overwrite the current state to hover
                    if new_pen_event.state != State.HOVER and State.HOVER in new_pen_event.state_history[
                                                                             -NUM_CHECK_LAST_EVENT_STATES:]:
                        print(
                            'Pen Event {} has prediction {}, but State.HOVER is present in the last {} events, so hover wins'.format(
                                last_pen_event.id, new_pen_event.state, NUM_CHECK_LAST_EVENT_STATES))
                        new_pen_event.state = State.HOVER

                new_pen_event.history = last_pen_event.history

                # Apply smoothing to the points by taking their previous positions into account
                # new_pen_event.x = int(SMOOTHING_FACTOR * (new_pen_event.x - last_pen_event.x) + last_pen_event.x)
                # new_pen_event.y = int(SMOOTHING_FACTOR * (new_pen_event.y - last_pen_event.y) + last_pen_event.y)

                new_pen_event.x = SMOOTHING_FACTOR * (new_pen_event.x - last_pen_event.x) + last_pen_event.x
                new_pen_event.y = SMOOTHING_FACTOR * (new_pen_event.y - last_pen_event.y) + last_pen_event.y

        elif len(self.active_pen_events) == 0 and len(new_pen_events) == 0:
            # print('Nothing to do')
            pass
        else:
            print('UNEXPECTED NUMBER OF PEN EVENTS!')

        if len(new_pen_events) > 1:
            print('WEIRD; too many new pen events')

        # Now we have al list of new events that need their own unique ID. Those are assigned now
        final_pen_events = self.assign_new_ids(new_pen_events)

        for final_pen_event in final_pen_events:
            # Add current position to the history list, but ignore hover events
            if final_pen_event.state != State.HOVER:
                final_pen_event.history.append((final_pen_event.x, final_pen_event.y))

            # num_total = len(final_pen_event.state_history)
            # num_hover = final_pen_event.state_history.count(State.HOVER)
            # num_draw = final_pen_event.state_history.count(State.DRAG)
            #
            # print('Hover: {}/{}; Draw: {}/{}'.format(num_hover, num_total, num_draw, num_total))

            # There needs to be at least one
            if len(final_pen_event.history) >= 1 and len(final_pen_event.state_history) >= NUM_HOVER_EVENTS_TO_END_LINE:
                # if final_pen_event.state == State.HOVER and State.HOVER not in final_pen_event.state_history[-3:]:
                # print(final_pen_event.state, final_pen_event.state_history[-4:])
                # if final_pen_event.state == State.HOVER and State.HOVER not in final_pen_event.state_history[-5:-1]:
                # num_hover_events = final_pen_event.state_history[-5:].count(State.HOVER)
                # num_draw_events = final_pen_event.state_history[-5:].count(State.DRAG)
                # if final_pen_event.state == State.HOVER and num_hover_events >= 2 and num_draw_events >= 2:
                # and final_pen_event.state_history[-2] == State.HOVER and final_pen_event.state_history[-3] == State.HOVER:
                # if final_pen_event.state_history[-1] == State.HOVER:

                # print(NUM_HOVER_EVENTS_TO_END_LINE, final_pen_event.state_history[-NUM_HOVER_EVENTS_TO_END_LINE:].count(State.HOVER))
                if final_pen_event.state_history[-NUM_HOVER_EVENTS_TO_END_LINE:].count(
                        State.HOVER) == NUM_HOVER_EVENTS_TO_END_LINE:
                    print('Pen Event {} turned from State.DRAG into State.HOVER'.format(final_pen_event.id))
                    if len(final_pen_event.history) > 0:
                        self.stored_lines.append(np.array(final_pen_event.history))
                    # print(final_pen_event.state, final_pen_event.state_history[-5:])
                    print('PEN Event {} ({} points) gets deleted because DRAW ended'.format(final_pen_event.id,
                                                                                            len(final_pen_event.history)))
                    final_pen_events = []

        return final_pen_events

    # Function to merge the events, even when multiple pens are present at the same time
    # def merge_pen_events(self, new_pen_events):
    #     now = round(time.time() * 1000)  # Get current timestamp
    #
    #     # # Keep track of the current state
    #     # # TODO: Do this already when the pen event is created
    #     # for new_pen_event in new_pen_events:
    #     #     new_pen_event.state_history = [new_pen_event.state]
    #
    #     # Iterate over copy of list
    #     # If a final_pen_event has been declared a "Click Event" in the last frame, this event is now over, and we can delete it.
    #     for active_pen_event in self.active_pen_events[:]:
    #         if active_pen_event.state == State.CLICK:
    #             self.process_click_events(active_pen_event)
    #
    #     # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
    #     shortest_distance_point_pairs = self.calculate_distances_between_all_points(self.active_pen_events,
    #                                                                                 new_pen_events, as_objects=True)
    #
    #     for entry in shortest_distance_point_pairs:
    #         last_pen_event = self.active_pen_events[entry[0]]
    #         new_pen_event = new_pen_events[entry[1]]
    #
    #         # We will reset the ID of all already paired events later. This check here will make sure that we do not
    #         # match an event multiple times
    #         if last_pen_event.id == -1:
    #             continue
    #
    #         if new_pen_event.state == State.HOVER and len(last_pen_event.history) > 0 and State.DRAG not in last_pen_event.state_history[:-3]:
    #             # print('No State.DRAG for at least 3 frames')
    #             pass
    #
    #         # TODO: Rework this check
    #         if new_pen_event.state == State.HOVER and State.HOVER not in last_pen_event.state_history[-3:]:
    #             # print('Pen Event {} turned from State.DRAG into State.HOVER'.format(last_pen_event.id))
    #             # new_pen_event.state_history.append(new_pen_event.state)
    #             # We now want to assign a new ID
    #             # TODO: Check why this event is called more than once
    #             # Maybe set state of old event to missing?
    #             continue
    #             # pass
    #
    #         new_pen_event.state_history = last_pen_event.state_history
    #         new_pen_event.state_history.append(new_pen_event.state)
    #
    #         # print(new_pen_event.state_history[-4:])
    #
    #         # Move ID and other important information from the active touch final_pen_event into the new
    #         # touch final_pen_event
    #         if State.HOVER in new_pen_event.state_history[-4:-2] and not State.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:  #  last_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
    #             pass
    #             # print('Pen Event {} turned from State.HOVER into State.DRAG'.format(last_pen_event.id))
    #
    #         if HOVER_WINS:
    #             # Overwrite the current state to hover
    #             if new_pen_event.state != State.HOVER and State.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:
    #                 # print('Pen Event {} has prediction {}, but State.HOVER is present in the last {} events, so hover wins'.format(last_pen_event.id, new_pen_event.state, KERNEL_SIZE_HOVER_WINS))
    #                 new_pen_event.state = State.HOVER
    #             else:
    #                 # print('Turning {} into a Drag event'.format(new_pen_event.state))
    #                 # if State.HOVER in new_pen_event.state_history[-4:-2]:
    #                 #     new_pen_event.state = State.NEW
    #                 # else:
    #                 # TODO: CHANGE this to allow for different types of drag events
    #                 # new_pen_event.state = State.DRAG  # last_pen_event.state
    #                 if new_pen_event.state != State.DRAG:
    #                     pass
    #
    #         # elif new_pen_event.state != State.HOVER:   # last_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
    #         #     print('HOVER EVENT turned into TOUCH EVENT')
    #         #     print('Check state history:', last_pen_event.state_history[-2:])
    #         #     if State.HOVER in last_pen_event.state_history[-2:]:
    #         #         print('Hover wins')
    #         #         new_pen_event.state_history.append(new_pen_event.state)
    #         #         new_pen_event.state = State.HOVER
    #         #     else:
    #         #         new_pen_event.state_history.append(new_pen_event.state)
    #         # else:
    #         #     new_pen_event.state = last_pen_event.state
    #         #     new_pen_event.state_history.append(new_pen_event.state)
    #
    #         new_pen_event.id = last_pen_event.id
    #         new_pen_event.first_appearance = last_pen_event.first_appearance
    #         new_pen_event.history = last_pen_event.history
    #
    #         # Apply smoothing to the points by taking their previous positions into account
    #         new_pen_event.x = int(SMOOTHING_FACTOR * (new_pen_event.x - last_pen_event.x) + last_pen_event.x)
    #         new_pen_event.y = int(SMOOTHING_FACTOR * (new_pen_event.y - last_pen_event.y) + last_pen_event.y)
    #
    #         # Set the ID of the last_pen_event back to -1 so that it is ignored in all future checks
    #         # We later want to only look at the remaining last_pen_event that did not have a corresponding new_pen_event
    #         last_pen_event.id = -1
    #
    #     # TODO: Maybe already do this earlier
    #     for new_pen_event in new_pen_events:
    #         new_pen_event.missing = False
    #         new_pen_event.last_seen_timestamp = now
    #
    #     # Check all active_pen_events that do not have a match found after comparison with the new_pen_events
    #     # It will be determined now if an event is over or not
    #     for active_pen_event in self.active_pen_events:
    #         # Skip all active_pen_events with ID -1. For those we already have found a match
    #         if active_pen_event.id == -1:
    #             continue
    #
    #         time_since_last_seen = now - active_pen_event.last_seen_timestamp
    #
    #         if not active_pen_event.missing or time_since_last_seen < TIME_POINT_MISSING_THRESHOLD_MS:
    #             if not active_pen_event.missing:
    #                 active_pen_event.last_seen_timestamp = now
    #
    #             active_pen_event.missing = True
    #             new_pen_events.append(active_pen_event)
    #
    #         else:
    #             # TODO: Rework these checks for our new approach
    #             if active_pen_event.state == State.NEW:
    #                 # We detected a click event, but we do not remove it yet because it also could be a double click.
    #                 # We will check this the next time this function is called.
    #                 # print('Click event candidate found')
    #                 active_pen_event.state = State.CLICK
    #                 new_pen_events.append(active_pen_event)
    #             elif active_pen_event.state == State.DRAG:
    #                 # End of a drag event
    #                 # print('DRAG Event ended for Pen Event {}'.format(active_pen_event.id))
    #                 self.pen_events_to_remove.append(active_pen_event)
    #                 # print('Adding {} points of Event {} to the stored_lines list'.format(len(active_pen_event.history),
    #                 #                                                                      active_pen_event.id))
    #                 self.stored_lines.append(np.array(active_pen_event.history))
    #                 # self.new_pen_events.append(active_pen_event.history)
    #             elif active_pen_event.state == State.HOVER:
    #                 # End of a Hover event
    #                 # print('HOVER Event ended for Pen Event {}'.format(active_pen_event.id))
    #                 self.pen_events_to_remove.append(active_pen_event)
    #
    #                 if len(active_pen_event.history) > 0:
    #                     # print('Adding {} points of Event {} to the stored_lines list'.format(
    #                     #     len(active_pen_event.history),
    #                     #     active_pen_event.id))
    #                     self.stored_lines.append(np.array(active_pen_event.history))
    #
    #
    #     # Now we have al list of new events that need their own unique ID. Those are assigned now
    #     final_pen_events = self.assign_new_ids(new_pen_events)
    #
    #     for final_pen_event in final_pen_events:
    #         # Add current position to the history list, but ignore hover events
    #         if final_pen_event.state != State.HOVER:
    #             final_pen_event.history.append((final_pen_event.x, final_pen_event.y))
    #
    #         # final_pen_event.state_history.append(final_pen_event.state)
    #
    #         time_since_first_appearance = now - final_pen_event.first_appearance
    #         if final_pen_event.state != State.CLICK and final_pen_event.state != State.DOUBLE_CLICK and time_since_first_appearance > CLICK_THRESH_MS:
    #             if final_pen_event.state == State.NEW:
    #                 # Start of a drag event
    #                 print('DRAG Event started for Pen Event {}'.format(final_pen_event.id))
    #                 final_pen_event.state = State.DRAG
    #             # elif final_pen_event.state == State.HOVER:
    #             #     print('DETECTED Hover EVENT!')
    #
    #     print(final_pen_events)
    #
    #     return final_pen_events

    def assign_new_ids(self, new_pen_events):
        final_pen_events = []

        for new_pen_event in new_pen_events:
            if new_pen_event.id == -1:
                new_pen_event.id = self.highest_id
                print('Assigned ID {} to a new Pen Event'.format(self.highest_id))
                self.highest_id += 1
            final_pen_events.append(new_pen_event)
        return final_pen_events

    def process_click_events(self, active_pen_event):
        # Check if click event happens without too much movement
        xs = []
        ys = []
        for x, y in active_pen_event.history:
            xs.append(x)
            ys.append(y)
        dx = abs(max(xs) - min(xs))
        dy = abs(max(ys) - min(ys))
        if dx < 5 and dy < 5:
            print('CLICK')
            # print('\a')

            # TODO: Add back double click events
            # # We have a new click event. Check if it belongs to a previous click event (-> Double click)
            # active_pen_event = self.check_if_double_click(now, active_pen_event, change_color)
            #
            # if active_pen_event.state == State.DOUBLE_CLICK:
            #     # Pass this point forward to the final return call because we want to send at least one alive
            #     # message for the double click event
            #     final_pen_events.append(active_pen_event)
            #
            #     # Give the Double Click event a different ID from the previous click event
            #     # active_pen_event.id = self.highest_id
            #     # self.highest_id += 1
            # else:
            #     # We now know that the current click event is no double click event,
            #     # but it might be the first click of a future double click. So we remember it.
            #     self.double_click_candidates.append(active_pen_event)

        self.pen_events_to_remove.append(active_pen_event)
        self.active_pen_events.remove(active_pen_event)

    def calculate_distances_between_all_points(self, point_list_one, point_list_two, as_objects=False):
        distances = []

        for i in range(len(point_list_one)):
            for j in range(len(point_list_two)):
                if as_objects:
                    distance_between_points = distance.euclidean(point_list_one[i].get_coordinates(),
                                                                 point_list_two[j].get_coordinates())
                else:
                    distance_between_points = distance.euclidean(point_list_one[i],
                                                                 point_list_two[j])

                if distance_between_points > MAX_DISTANCE_FOR_MERGE:
                    print('DISTANCE {} TOO LARGE'.format(distance_between_points))
                    continue
                distances.append([i, j, distance_between_points])

        # Sort list of lists by third element, in this case the distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        distances.sort(key=lambda x: x[2])

        return distances


class IRPenDebugger:
    frame_counter = 0
    start_time = 0
    rois = []

    active_pen_events = []

    uds_initialized = False

    def __init__(self):
        from flir_blackfly_s import FlirBlackflyS

        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        self.analogue_digital_document = AnalogueDigitalDocumentsDemo()

        self.logitech_brio_camera = LogitechBrio(self)
        self.logitech_brio_camera.init_video_capture()
        self.logitech_brio_camera.start()

        thread = threading.Thread(target=self.debug_mode_thread)
        thread.start()

    current_id = -1
    def on_new_brio_frame(self, frame, homography_matrix):
        # print(frame.shape)

        highlight_dict, highlights_removed = self.analogue_digital_document.get_highlight_rectangles(frame, homography_matrix)

        if highlights_removed:
            self.clear_rects()

        for highlight_id, rectangle in highlight_dict.items():
            # TODO: Send info for highlights that should be removed
            self.send_rect(highlight_id, 1, rectangle)

        # print('Final rectangles:', highlight_rectangles)

    # id: id of the rect (writing to an existing id should move the rect)
    # state: alive = 1, dead = 0; use it to remove unused rects!
    # coords list: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] -- use exactly four entries and make sure they are sorted!
    def send_rect(self, id, state, coord_list):
        if not self.uds_initialized:
            return 0

        message = f'r {id} '
        for coords in coord_list:
            # message += f'{coords[0]} {coords[1]}'
            message += f'{coords} '
        message += f'{state} '

        # print('message', message)

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            self.sock.send(message.encode())

        return 1

    def clear_rects(self):
        if not self.uds_initialized:
            return 0
        if ENABLE_UNIX_SOCKET:
            self.sock.send('c '.encode())
        return 1

    def debug_mode_thread(self):
        if ENABLE_FIFO_PIPE:
            pipe_name = 'pipe_test'
            if not os.path.exists(pipe_name):
                os.mkfifo(pipe_name)
            self.pipeout = os.open(pipe_name, os.O_WRONLY)

        if ENABLE_UNIX_SOCKET:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            print('connecting to %s' % UNIX_SOCK_NAME)
            try:
                self.sock.connect(UNIX_SOCK_NAME)
            except socket.error as msg:
                print(msg)
                sys.exit(1)
            self.uds_initialized = True

        if DEBUG_MODE:
            cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ROI', 1000, 500)

        while True:
            if SEND_TO_FRONTEND:
                if len(self.active_pen_events) > 0:
                    # if len(self.active_pen_events[0].history) > 0:
                    message = 'l {} {} {} {} '.format(self.active_pen_events[0].id,
                                                      int(self.active_pen_events[0].x),
                                                      int(self.active_pen_events[0].y),
                                                      0 if self.active_pen_events[0].state == State.HOVER else 1)
                    # print(message)
                    if ENABLE_FIFO_PIPE:
                        os.write(self.pipeout, bytes(message, 'utf8'))
                    if ENABLE_UNIX_SOCKET:
                        self.sock.send(message.encode())
                    self.active_pen_events = []
                if not DEBUG_MODE:
                    # TODO: CHeck why we need this here. time.sleep() does not work here
                    key = cv2.waitKey(1)

            if DEBUG_MODE:
                if len(self.rois) == 2:
                    roi0 = cv2.resize(self.rois[0], (500, 500), interpolation=cv2.INTER_AREA)
                    max0 = str(np.max(roi0))
                    roi1 = cv2.resize(self.rois[1], (500, 500), interpolation=cv2.INTER_AREA)
                    max1 = str(np.max(roi1))

                    roi0 = cv2.putText(
                        img=roi0,
                        text=max0,
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=3.0,
                        color=(255),
                        thickness=3
                    )

                    roi1 = cv2.putText(
                        img=roi1,
                        text=max1,
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=3.0,
                        color=(255),
                        thickness=3
                    )

                    cv2.imshow('ROI', cv2.hconcat([roi0, roi1]))
                    self.rois = []
                elif len(self.rois) == 1:
                    roi0 = cv2.resize(self.rois[0], (500, 500), interpolation=cv2.INTER_AREA)
                    max0 = str(np.max(roi0))

                    roi0 = cv2.putText(
                        img=roi0,
                        text=max0,
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=3.0,
                        color=(255),
                        thickness=3
                    )
                    roi1 = np.zeros((500, 500), np.uint8)
                    cv2.imshow('ROI', cv2.hconcat([roi0, roi1]))
                    self.rois = []

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    sys.exit(0)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(frames) > 0:
            self.frame_counter += 1
            active_pen_events, stored_lines, _, _, debug_distances, rois = self.ir_pen.get_ir_pen_events_multicam(
                frames, matrices)
            self.rois = rois
            self.active_pen_events = active_pen_events

            self.analogue_digital_document.on_new_finished_line(stored_lines)


if __name__ == '__main__':
    debugger = IRPenDebugger()
