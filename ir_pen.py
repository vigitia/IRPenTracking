import sys
from enum import Enum
import time
import numpy as np
from tensorflow import keras
from tflite import LiteModel
import datetime
from scipy.spatial import distance
from skimage.feature import peak_local_max

from cv2 import cv2

# Aktuell bestes model:
# MODEL_PATH = 'evaluation/hover_predictor_flir_2'

MODEL_PATH = 'evaluation/hover_predictor_flir_8'

CROP_IMAGE_SIZE = 48

# Simple Smoothing
SMOOTHING_FACTOR = 0.8  # Value between 0 and 1, depending on if the old or the new value should count more.


# Amount of time a point can be missing until the event "on click/drag stop" will be fired
TIME_POINT_MISSING_THRESHOLD_MS = 22

# Point needs to appear and disappear within this timeframe in ms to count as a click (vs. a drag event)
CLICK_THRESH_MS = 10

# Hover will be selected over Draw if Hover Event is within the last X event states
KERNEL_SIZE_HOVER_WINS = 2

MIN_BRIGHTNESS_FOR_PREDICTION = 50

MAX_DISTANCE_DRAW = 50

# Removes all points that are not within the screen coordinates (e.g. 3840x2160)
REMOVE_EVENTS_OUTSIDE_BORDER = True # TODO!


DEBUG_MODE = False

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

CAMERA_WIDTH = 1920  # 848
CAMERA_HEIGHT = 1200  # 480

STATES = ['draw', 'hover', 'undefined']

TRAINING_DATA_COLLECTION_MODE = False
TRAIN_STATE = 'hover_close_2_500'
TRAIN_PATH = 'out3/2022-07-15'
TRAIN_IMAGE_COUNT = 3000


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            # print("I " + prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            # print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) + " ms)")
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return retval
        return wrapper
    return timeit_decorator


# Enum to define State of a Point
class State(Enum):
    NEW = -1  # A new event where it is not yet sure if it will just be a click event or a drag event
    CLICK = 0
    DRAG = 1
    DOUBLE_CLICK = 2
    HOVER = 3


class PenEvent:

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.id = -1

        self.missing = False
        self.last_seen_timestamp = 0

        self.first_appearance = round(time.time() * 1000)
        self.state = State.NEW
        self.history = []  # All logged x and y positions as tuples
        self.state_history = []  # All logged states (Hover, draw, ...)

        self.alive = True  # A pen event is alive if it is not missing

    def get_coordinates(self):
        return tuple([self.x, self.y])

    def __repr__(self):
        return 'Pen Event {} at ({}, {}). Type: {}. Num Points: {}'.format(str(self.id), str(self.x), str(self.y),
                                                                           self.state, len(self.history))


class IRPen:

    active_pen_events = []
    new_pen_events = []

    highest_id = 1  # Assign each new pen event a new id. This variable keeps track of the highest number.

    # If a pen event is over, it will be deleted. But its points will be stored here to keep track of all drawn lines.
    stored_lines = []
    pen_events_to_remove = []  # Points that got deleted from active_points in the current frame

    double_click_candidates = []

    # Only needed during training TRAINING_DATA_COLLECTION_MODE. Keeps track of the number of saved images
    saved_image_counter = 0

    def __init__(self):
        # Init Keras
        keras.backend.clear_session()
        self.keras_lite_model = LiteModel.from_keras_model(keras.models.load_model(MODEL_PATH))

    def save_training_image(self, img, pos):
        if self.saved_image_counter == 0:
            print('Starting in 10 Seconds')
            time.sleep(10)

        self.saved_image_counter += 1
        if self.saved_image_counter % 10 == 0:
            cv2.imwrite(f'{TRAIN_PATH}/{TRAIN_STATE}/{TRAIN_STATE}_{int(self.saved_image_counter / 10)}_{pos[0]}_{pos[1]}.png', img)
            print(f'saving frame {int(self.saved_image_counter / 10)}/{TRAIN_IMAGE_COUNT}')

        if self.saved_image_counter / 10 >= TRAIN_IMAGE_COUNT:
            sys.exit(0)

    def transform_coords_to_output_res(self, x, y, transform_matrix):
        coords = np.array([x, y, 1])

        transformed_coords = transform_matrix.dot(coords)
        # Normalize coordinates by dividing by z
        transformed_coords = (int(transformed_coords[0] / transformed_coords[2]),
                              int(transformed_coords[1] / transformed_coords[2]))
        return transformed_coords

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

        debug_distances = [0, (0, 0)]

        # WIP new approach for multiple pens
        WIP_APPROACH_MULTI_PEN = False
        if WIP_APPROACH_MULTI_PEN:
            for i, frame in enumerate(camera_frames):

                # Add a new sublist for each frame
                predictions.append([])
                rois.append([])
                roi_coords.append([])
                brightness_values.append([])
                subpixel_coords.append([])

                rois_new, roi_coords_new, max_brightness_values = self.get_all_rois(frame)

                for j, pen_event_roi in enumerate(rois_new):

                    if TRAINING_DATA_COLLECTION_MODE:
                        self.save_training_image(pen_event_roi, (roi_coords_new[j][0], roi_coords_new[j][1]))
                        continue

                    prediction, confidence = self.predict(pen_event_roi)

                    predictions[i].append(prediction)
                    rois[i].append(pen_event_roi)

                    transformed_coords = self.transform_coords_to_output_res(roi_coords_new[j][0], roi_coords_new[j][1], transform_matrices[i])
                    roi_coords[i].append(transformed_coords)

                    brightness_values[i].append(max_brightness_values[j])

                    (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)
                    subpixel_coords[i].append((x, y))

            new_pen_events = self.generate_new_pen_events(subpixel_coords, predictions, brightness_values)

            # This function needs to be called even if there are no new pen events to update all existing events
            self.active_pen_events = self.merge_pen_events(new_pen_events)

            return self.active_pen_events, self.stored_lines, self.new_pen_events, self.pen_events_to_remove, debug_distances

        # 8 ms
        for i, frame in enumerate(camera_frames):
            # TODO: Get here all spots and not just one

            # crop 1: 0.5 ms
            # crop 2: 0.5 - 1 ms (Ausreißer bis 6 ms)
            pen_event_roi, brightest, (x, y) = self.crop_image(frame)

            if TRAINING_DATA_COLLECTION_MODE:
                self.save_training_image(pen_event_roi, (x, y))
                continue

            if brightest > MIN_BRIGHTNESS_FOR_PREDICTION and pen_event_roi.shape[0] == CROP_IMAGE_SIZE and pen_event_roi.shape[1] == CROP_IMAGE_SIZE:
                rois.append(pen_event_roi)

                transformed_coords = self.transform_coords_to_output_res(x, y, transform_matrices[i])
                roi_coords.append(transformed_coords)

                brightness_values.append(brightest)

                (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)

                subpixel_coords.append((x, y))
                debug_distances.append(subpixel_coords)

        # If we see only one point:
        if len(subpixel_coords) == 1:
            prediction, confidence = self.predict(rois[0])
            # print(prediction)
            if prediction == 'draw':
                # print('Status: Touch')
                new_ir_pen_event = PenEvent(subpixel_coords[0][0], subpixel_coords[0][1])
                new_ir_pen_event.state = State.DRAG
                new_pen_events.append(new_ir_pen_event)

            elif prediction == 'hover':
                # print('Status: Hover')
                new_ir_pen_event = PenEvent(subpixel_coords[0][0], subpixel_coords[0][1])
                new_ir_pen_event.state = State.HOVER
                new_pen_events.append(new_ir_pen_event)
            else:
                print('Unknown state')

        # If we see two points
        elif len(subpixel_coords) == 2:
            distance_between_points = distance.euclidean(subpixel_coords[0],
                                                         subpixel_coords[1])

            (center_x, center_y) = self.get_center(subpixel_coords[0], subpixel_coords[1])
            debug_distances[0] = int(distance_between_points)
            debug_distances[1] = (center_x, center_y)

            # print(distance_between_points, center_x, center_y)

            if distance_between_points > MAX_DISTANCE_DRAW:
                # Calculate center between the two points

                new_ir_pen_event = PenEvent(center_x, center_y)
                new_ir_pen_event.state = State.HOVER
                new_pen_events.append(new_ir_pen_event)
            else:
                for pen_event_roi in rois:
                    prediction, confidence = self.predict(pen_event_roi)
                    predictions.append(prediction)

                if all(x == predictions[0] for x in predictions):
                    # The predictions for all cameras are the same
                    final_prediction = predictions[0]
                else:
                    brightest_image_index = brightness_values.index(max(brightness_values))
                    # There is a disagreement
                    # Currently we then use the prediction of the brightest point in all camera frames
                    final_prediction = predictions[brightest_image_index]
                    # TODO: OR HOVER WINS HERE

                if final_prediction == 'draw':
                    # print('Status: Touch')
                    new_ir_pen_event = PenEvent(center_x, center_y)
                    new_ir_pen_event.state = State.DRAG
                    new_pen_events.append(new_ir_pen_event)

                elif final_prediction == 'hover':
                    # print('Status: Hover')
                    new_ir_pen_event = PenEvent(center_x, center_y)
                    new_ir_pen_event.state = State.HOVER
                    new_pen_events.append(new_ir_pen_event)
                else:
                    print('Unknown state')

        # This function needs to be called even if there are no new pen events to update all existing events
        self.active_pen_events = self.merge_pen_events(new_pen_events)

        return self.active_pen_events, self.stored_lines, self.new_pen_events, self.pen_events_to_remove, debug_distances

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

        print(' ')
        # print(subpixel_coords)

        # Deal with single points first
        if len(subpixel_coords[0]) == 0 and len(subpixel_coords[1]) > 0 or len(subpixel_coords[0]) > 0 and len(subpixel_coords[1]) == 0:
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

                        (center_x, center_y) = self.get_center(subpixel_coords[0][entry[0]], subpixel_coords[1][entry[1]])
                        cv2.line(zeros, subpixel_coords[0][entry[0]], subpixel_coords[1][entry[1]], (255, 255, 255), 3)
                        used_left.append(entry[0])
                        used_right.append(entry[1])

                        # TODO: Compare brightness and use the prediction of the brighter point
                        prediction_a = predictions[0][entry[0]]
                        prediction_b = predictions[1][entry[1]]

                        new_pen_events.append(self.generate_new_pen_event(prediction_a, center_x,center_y))

            for i in range(len(subpixel_coords[0])):
                if i not in used_left:
                    print('Point {} on cam 0 remains'.format(i))
                    new_pen_events.append(self.generate_new_pen_event(predictions[0][i], subpixel_coords[0][i][0], subpixel_coords[0][i][1]))

            for i in range(len(subpixel_coords[1])):
                if i not in used_right:
                    print('Point {} on cam 1 remains'.format(i))
                    new_pen_events.append(self.generate_new_pen_event(predictions[1][i], subpixel_coords[1][i][0],
                                                                      subpixel_coords[1][i][1]))



            #print(shortest_distance_point_pairs)
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
            new_ir_pen_event = PenEvent(x, y)
            new_ir_pen_event.state = State.DRAG
            return new_ir_pen_event

        elif prediction == 'hover':
            # print('Status: Hover')
            new_ir_pen_event = PenEvent(x, y)
            new_ir_pen_event.state = State.HOVER
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

    # @timeit('All ROIs')
    # TOO SLOW :(
    def get_all_rois_bad(self, img, size=CROP_IMAGE_SIZE):
        start_time = datetime.datetime.now()
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        end_time = datetime.datetime.now()
        run_time = (end_time - start_time).microseconds / 1000.0
        print('RUN TIME:', run_time)
        peaks = peak_local_max(img, min_distance=10, threshold_abs=50)

        rois = []
        roi_coords = []
        margin = int(size / 2)
        max_brightness_values = []

        for point in peaks:
            x = point[1]
            y = point[0]

            # Dead pixel fix
            if x == 46 and y == 565:
                # TODO: Find solution for dead pixel
                continue

            roi = img[y - margin: y + margin, x - margin: x + margin]
            if roi.shape[0] != size or roi.shape[1] != size:
                print('WRONG SHAPE')
                # TODO: FIX THIS
                return rois, roi_coords, max_brightness_values
            rois.append(roi)
            roi_coords.append((x, y))
            _, brightest, _, _ = cv2.minMaxLoc(roi)
            max_brightness_values.append(int(brightest))

        return rois, roi_coords, max_brightness_values

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
        print('using crop_image_2() function')
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

        img = img.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 1)
        prediction = self.keras_lite_model.predict(img)
        if not prediction.any():
            return STATES[-1], 0
        state = STATES[np.argmax(prediction)]
        confidence = np.max(prediction)
        return state, confidence

    # TODO: Offset fixen
    def find_pen_position_subpixel_crop(self, image, center_original):
        w = image.shape[0]
        h = image.shape[1]
        # print('1', ir_image.shape)
        #center_original = (coords_original[0] + w/2, coords_original[1] + h/2)

        factor_w = WINDOW_WIDTH / CAMERA_WIDTH
        factor_h = WINDOW_HEIGHT / CAMERA_HEIGHT
        new_w = int(w * factor_w)
        new_h = int(h * factor_h)
        top_left_scaled = (center_original[0] * factor_w - new_w / 2, center_original[1] * factor_h - new_h / 2)

        if len(image.shape) == 3:
            ir_image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            ir_image_grey = image
        # TODO:
        # print('2', ir_image_grey.shape)
        _, thresh = cv2.threshold(ir_image_grey, np.max(ir_image_grey) - 1, 255, cv2.THRESH_BINARY)


        # TODO: resize only cropped area
        thresh_large = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        min_radius = ir_image_grey.shape[0]
        smallest_contour = contours[0]

        # print(len(contours))
        # Find the smallest contour if there are multiple (we want to find the pen tip, not its light beam
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                smallest_contour = contour

        # Find the center of the contour using OpenCV Moments
        M = cv2.moments(smallest_contour)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        position = (int(top_left_scaled[0] + cX), int(top_left_scaled[1] + cY))

        return position, min_radius

    # 3 - 5 ms

    def find_pen_position_subpixel_old(self, ir_image):

        if len(ir_image.shape) == 3:
            ir_image_grey = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
        else:
            ir_image_grey = ir_image
        _, thresh = cv2.threshold(ir_image_grey, np.max(ir_image_grey) - 1, 255, cv2.THRESH_BINARY)

        # TODO: resize only cropped area
        thresh_large = cv2.resize(thresh, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)

        contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]
        min_radius = ir_image_grey.shape[0]
        smallest_contour = contours[0]

        # print(len(contours))
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                smallest_contour = contour

        # Find the center of the contour using OpenCV Moments
        M = cv2.moments(smallest_contour)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        position = (cX, cY)

        return position, min_radius

        # left_ir_image_large = cv2.resize(left_ir_image.copy(), (3840, 2160), interpolation=cv2.INTER_AREA)
        # cv2.circle(left_ir_image_large, position, 1, (0, 0, 0))
        # cv2.imshow('large', left_ir_image_large)

    # Finds the exact center
    def find_pen_position_old(self, img):
        _, thresh = cv2.threshold(img, np.max(img) - 30, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        position = (0, 0)
        min_radius = img.shape[0]
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                position = (int(x), int(y))
        return min_radius, position

    def find_pen_orientation_old(self, image):
        (image_height, image_width) = image.shape
        step_w = int(image_width / 10)
        step_h = int(image_height / 10)

        samples = []
        for x in range(3 * step_w, (image_width - 3 * step_w), step_w):
            for y in range(3 * step_h, (image_height - 3 * step_h), step_h):
                samples.append(image[y, x])
                
        _, thresh = cv2.threshold(image, np.median(samples) * 1.2, 255, cv2.THRESH_BINARY)
        
        #y_nonzero, x_nonzero = np.nonzero(image)
        #x1 = np.min(x_nonzero)
        #x2 = np.max(x_nonzero)
        #y1 = np.min(y_nonzero)
        #y2 = np.max(y_nonzero)
        
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            return -1, -1

        contours = contours[0] if len(contours) == 2 else contours[1]

        #cntr = contours[0]
        #x,y,w,h = cv2.boundingRect(cntr)
        #hull = cv2.convexHull(contours[0])
        
        [vx, vy, x, y] = cv2.fitLine(contours, cv2.DIST_L2, 0, 0.01, 0.01)
        
        rows, cols = image.shape[:2]
        y_left = int((-x*vy/vx) + y)
        y_right = int(((cols-x)*vy/vx)+y)
        return y_left, y_right

    def merge_pen_events(self, new_pen_events):
        now = round(time.time() * 1000)  # Get current timestamp

        # Keep track of the current state
        # TODO: Do this already when the pen event is created
        for new_pen_event in new_pen_events:
            new_pen_event.state_history = [new_pen_event.state]

        # Iterate over copy of list
        # If a final_pen_event has been declared a "Click Event" in the last frame, this event is now over, and we can delete it.
        for active_pen_event in self.active_pen_events[:]:
            if active_pen_event.state == State.CLICK:
                self.process_click_events(active_pen_event)

        # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
        shortest_distance_point_pairs = self.calculate_distances_between_all_points(self.active_pen_events,
                                                                                    new_pen_events, as_objects=True)

        for entry in shortest_distance_point_pairs:
            last_pen_event = self.active_pen_events[entry[0]]
            new_pen_event = new_pen_events[entry[1]]

            # We will reset the ID of all already paired events later. This check here will make sure that we do not
            # match an event multiple times
            if last_pen_event.id == -1:
                continue

            # TODO: Rework this check
            if new_pen_event.state == State.HOVER and State.HOVER not in last_pen_event.state_history[-3:]:
                print('TOUCH EVENT turned into HOVER EVENT')
                # new_pen_event.state_history.append(new_pen_event.state)
                # We now want to assign a new ID
                # TODO: Check why this event is called more than once
                continue

            new_pen_event.state_history = last_pen_event.state_history
            new_pen_event.state_history.append(new_pen_event.state)

            # print(new_pen_event.state_history[-4:])

            # Move ID and other important information from the active touch final_pen_event into the new
            # touch final_pen_event
            if State.HOVER in new_pen_event.state_history[-4:-2] and not State.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:  #  last_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
                pass
                # print('HOVER EVENT turned into TOUCH EVENT')

            # Overwrite the current state to hover
            if State.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:
                print('Hover wins')
                new_pen_event.state = State.HOVER
            else:
                print('Turning {} into a Drag event'.format(new_pen_event.state))
                # if State.HOVER in new_pen_event.state_history[-4:-2]:
                #     new_pen_event.state = State.NEW
                # else:
                # TODO: CHANGE this to allow for different types of drag events
                new_pen_event.state = State.DRAG  # last_pen_event.state

            # elif new_pen_event.state != State.HOVER:   # last_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
            #     print('HOVER EVENT turned into TOUCH EVENT')
            #     print('Check state history:', last_pen_event.state_history[-2:])
            #     if State.HOVER in last_pen_event.state_history[-2:]:
            #         print('Hover wins')
            #         new_pen_event.state_history.append(new_pen_event.state)
            #         new_pen_event.state = State.HOVER
            #     else:
            #         new_pen_event.state_history.append(new_pen_event.state)
            # else:
            #     new_pen_event.state = last_pen_event.state
            #     new_pen_event.state_history.append(new_pen_event.state)

            new_pen_event.id = last_pen_event.id
            new_pen_event.first_appearance = last_pen_event.first_appearance
            new_pen_event.history = last_pen_event.history

            # Apply smoothing to the points by taking their previous positions into account
            new_pen_event.x = int(SMOOTHING_FACTOR * (new_pen_event.x - last_pen_event.x) + last_pen_event.x)
            new_pen_event.y = int(SMOOTHING_FACTOR * (new_pen_event.y - last_pen_event.y) + last_pen_event.y)

            # Set the ID of the last_pen_event back to -1 so that it is ignored in all future checks
            # We later want to only look at the remaining last_pen_event that did not have a corresponding new_pen_event
            last_pen_event.id = -1

        # TODO: Maybe already do this earlier
        for new_pen_event in new_pen_events:
            new_pen_event.missing = False
            new_pen_event.last_seen_timestamp = now

        # Check all active_pen_events that do not have a match found after comparison with the new_pen_events
        # It will be determined now if an event is over or not
        for active_pen_event in self.active_pen_events:
            # Skip all active_pen_events with ID -1. For those we already have found a match
            if active_pen_event.id == -1:
                continue

            time_since_last_seen = now - active_pen_event.last_seen_timestamp

            if not active_pen_event.missing or time_since_last_seen < TIME_POINT_MISSING_THRESHOLD_MS:
                if not active_pen_event.missing:
                    active_pen_event.last_seen_timestamp = now

                active_pen_event.missing = True
                new_pen_events.append(active_pen_event)

            else:
                # TODO: Rework these checks for our new approach
                if active_pen_event.state == State.NEW:
                    # We detected a click event, but we do not remove it yet because it also could be a double click.
                    # We will check this the next time this function is called.
                    print('Click event candidate found')
                    active_pen_event.state = State.CLICK
                    new_pen_events.append(active_pen_event)
                elif active_pen_event.state == State.DRAG:
                    # End of a drag event
                    print('DRAG END')
                    self.pen_events_to_remove.append(active_pen_event)
                    self.stored_lines.append(np.array(active_pen_event.history))
                    self.new_pen_events.append(active_pen_event.history)
                elif active_pen_event.state == State.HOVER:
                    # End of a Hover event
                    print('HOVER EVENT END')
                    self.pen_events_to_remove.append(active_pen_event)

        # Now we have al list of new events that need their own unique ID. Those are assigned now
        final_pen_events = self.assign_new_ids(new_pen_events)

        for final_pen_event in final_pen_events:
            # Add current position to the history list, but ignore hover events
            if final_pen_event.state != State.HOVER:
                final_pen_event.history.append((final_pen_event.x, final_pen_event.y))

            # final_pen_event.state_history.append(final_pen_event.state)

            time_since_first_appearance = now - final_pen_event.first_appearance
            if final_pen_event.state != State.CLICK and final_pen_event.state != State.DOUBLE_CLICK and time_since_first_appearance > CLICK_THRESH_MS:
                if final_pen_event.state == State.NEW:
                    # Start of a drag event
                    print('DRAG START')
                    final_pen_event.state = State.DRAG
                # elif final_pen_event.state == State.HOVER:
                #     print('DETECTED Hover EVENT!')

        return final_pen_events

    def assign_new_ids(self, new_pen_events):
        final_pen_events = []

        for new_pen_event in new_pen_events:
            if new_pen_event.id == -1:
                new_pen_event.id = self.highest_id
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
                distances.append([i, j, distance_between_points])

        # Sort list of lists by third element, in this case the distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        distances.sort(key=lambda x: x[2])

        return distances


if __name__ == '__main__':
    from flir_blackfly_s import FlirBlackflyS
    flir_blackfly_s = FlirBlackflyS()
    flir_blackfly_s.start()

    ir_pen = IRPen()

    while True:
        new_frames, matrices = flir_blackfly_s.get_camera_frames()

        if len(new_frames) > 0:
            active_pen_events, stored_lines, _, _, debug_distances = ir_pen.get_ir_pen_events_multicam(new_frames,
                                                                                                       matrices)

            print(active_pen_events)
