
import sys
import time
import datetime
import numpy as np
import skimage

from scipy.spatial import distance
from cv2 import cv2

from pen_state import PenState
from pen_event import PenEvent
from pen_events_controller import PenEventsController
from ir_pen_cnn import IRPenCNN


CROP_IMAGE_SIZE = 48  # Currently 48x48 Pixel

MIN_BRIGHTNESS_FOR_PREDICTION = 50  # A spot in the camera image needs to have at least X brightness to be considered.

USE_MAX_DISTANCE_DRAW = False
MAX_DISTANCE_DRAW = 500  # Maximum allowed pixel distance between two points to be considered the same line ID

LATENCY_MEASURING_MODE = False

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

CAMERA_WIDTH = 1920  # 848
CAMERA_HEIGHT = 1200  # 480

DEBUG_MODE = False  # Enable for Debug print statements and preview windows


# For debugging purposes
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


class IRPen:

    factor_width = WINDOW_WIDTH / CAMERA_WIDTH
    factor_height = WINDOW_HEIGHT / CAMERA_HEIGHT

    # active_learning_counter = 0
    # active_learning_state = 'hover'

    # last_coords = (0, 0)
    # last_frame_time = time.time()

    def __init__(self):
        self.ir_pen_cnn = IRPenCNN()
        self.pen_events_controller = PenEventsController()

    # @timeit('Pen Events')
    def get_ir_pen_events(self, camera_frames, transform_matrices):

        new_pen_events = []

        predictions = []

        rois = []
        transformed_roi_coords = []
        brightness_values = []
        # subpixel_precision_coords = []

        for i, frame in enumerate(camera_frames):

            # TODO: Get here all spots and not just one
            pen_event_roi, brightest, (x, y) = self.crop_image(frame)

            if pen_event_roi is not None:
                transformed_coords = self.transform_coords_to_output_res(x, y, transform_matrices[i])
                # (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)

                rois.append(pen_event_roi)
                transformed_roi_coords.append(transformed_coords)
                brightness_values.append(np.sum(pen_event_roi))
                # subpixel_precision_coords.append((x, y))

        # TODO: TESTING ONLY! We currently do not call the find_pen_position_subpixel_crop() function
        subpixel_precision_coords = transformed_roi_coords

        # If we see only one point:
        if len(subpixel_precision_coords) == 1:
            prediction, confidence = self.ir_pen_cnn.predict(rois[0])
            # print('One Point', prediction, confidence)
            # TODO: Also consider confidence here
            if prediction == 'draw':
                # print('One point draw')
                new_ir_pen_event = PenEvent(subpixel_precision_coords[0][0], subpixel_precision_coords[0][1], PenState.DRAG)
                new_pen_events.append(new_ir_pen_event)

            elif prediction == 'hover':
                # print('One point hover')
                new_ir_pen_event = PenEvent(subpixel_precision_coords[0][0], subpixel_precision_coords[0][1], PenState.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                print('Error: Unknown state')

        # If we see two points
        elif len(subpixel_precision_coords) == 2:
            # print('Two points')

            # Calculate a new point that is the center between the two given points
            (center_x, center_y) = self.get_center_point(subpixel_precision_coords[0], subpixel_precision_coords[1])

            distance_between_points = distance.euclidean(subpixel_precision_coords[0], subpixel_precision_coords[1])
            # print('DISTANCE:', distance_between_points, flush=True)

            if USE_MAX_DISTANCE_DRAW and distance_between_points > MAX_DISTANCE_DRAW:
                print('Distance too large -> Hover', distance_between_points)
                # Calculate center between the two points

                final_prediction = 'hover'
            else:
                for pen_event_roi in rois:
                    prediction, confidence = self.ir_pen_cnn.predict(pen_event_roi)
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
                new_ir_pen_event = PenEvent(center_x, center_y, PenState.DRAG)
                new_pen_events.append(new_ir_pen_event)

            elif final_prediction == 'hover':
                # print('Status: Hover')
                new_ir_pen_event = PenEvent(center_x, center_y, PenState.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                print('Error: Unknown state')
                time.sleep(30)

        if LATENCY_MEASURING_MODE:
            if len(subpixel_precision_coords) == 0:
                new_ir_pen_event = PenEvent(0, 0, PenState.HOVER)
                new_pen_events.append(new_ir_pen_event)
            else:
                new_pen_events[-1] = PenEvent(center_x, center_y, PenState.DRAG)

        # For distance/velocity calculation
        # current_frame_time = time.time()
        # delta_time = current_frame_time - self.last_frame_time
        # dist = distance.euclidean((center_x, center_y),
        #                           self.last_coords)
        # self.last_coords = (center_x, center_y)
        # self.last_frame_time = current_frame_time
        # print('LOG {}, {}'.format(distance_between_points, abs(dist) / delta_time), flush=True)

        # This function needs to be called even if there are no new pen events to update all existing events
        active_pen_events, stored_lines, pen_events_to_remove = self.pen_events_controller.merge_pen_events_single(new_pen_events)

        # TODO: REWORK RETURN. stored_lines not always needed
        return active_pen_events, stored_lines, new_pen_events, pen_events_to_remove, rois

    # TODO: Complete this approach
    # WIP!
    def get_ir_pen_events_new(self, camera_frames, transform_matrices):

        brightness_values = []

        predictions = []
        rois = []
        roi_coords = []
        subpixel_coords = []

        debug_distances = []

        for i, frame in enumerate(camera_frames):

            # Add a new sublist for each frame
            predictions.append([])
            rois.append([])
            roi_coords.append([])
            brightness_values.append([])
            subpixel_coords.append([])

            rois_new, roi_coords_new, max_brightness_values = self.get_all_rois(frame)

            for j, pen_event_roi in enumerate(rois_new):

                prediction, confidence = self.ir_pen_cnn.predict(pen_event_roi)

                predictions[i].append(prediction)
                rois[i].append(pen_event_roi)

                transformed_coords = self.transform_coords_to_output_res(roi_coords_new[j][0], roi_coords_new[j][1], transform_matrices[i])
                roi_coords[i].append(transformed_coords)

                brightness_values[i].append(max_brightness_values[j])

                (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)
                subpixel_coords[i].append((x, y))

        new_pen_events = self.generate_new_pen_events(subpixel_coords, predictions, brightness_values)

        # This function needs to be called even if there are no new pen events to update all existing events
        active_pen_events, stored_lines, pen_events_to_remove = self.pen_events_controller.merge_pen_events_single(
            new_pen_events)

        # TODO: REWORK RETURN. stored_lines not always needed
        return active_pen_events, stored_lines, new_pen_events, pen_events_to_remove, debug_distances, rois

    # By using the transform matrix handed over by the camera, the given coordinate will be transformed from camera
    # space to projector space
    def transform_coords_to_output_res(self, x, y, transform_matrix):
        try:
            coords = np.array([x, y, 1])

            transformed_coords = transform_matrix.dot(coords)

            # Normalize coordinates by dividing by z
            transformed_coords = (transformed_coords[0] / transformed_coords[2],
                                  transformed_coords[1] / transformed_coords[2])

            # Coordinates are now aligned with the projection area but still need to be upscaled to the output resolution
            transformed_coords = (transformed_coords[0] * self.factor_width, transformed_coords[1] * self.factor_height)

            return transformed_coords
        except Exception as e:
            print(e)
            print('Error in transform_coords_to_output_res(). Maybe the transform_matrix is malformed?')
            print('This error could also appear if CALIBRATION_MODE is still enabled in flir_blackfly_s.py')
            time.sleep(5)
            sys.exit(1)

    # WIP!
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

                        (center_x, center_y) = self.get_center_point(subpixel_coords[0][entry[0]],
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
            new_ir_pen_event = PenEvent(x, y, PenState.DRAG)
            # new_ir_pen_event.state = State.DRAG
            return new_ir_pen_event

        elif prediction == 'hover':
            # print('Status: Hover')
            new_ir_pen_event = PenEvent(x, y, PenState.HOVER)
            # new_ir_pen_event.state = State.HOVER
            return new_ir_pen_event
        else:
            print('Error: Unknown state')
            sys.exit(1)

    # Calculate the center point between two given points
    def get_center_point(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2

        x_dist = abs(x1 - x2) / 2
        y_dist = abs(y1 - y2) / 2
        return min(x1, x2) + x_dist, min(y1, y2) + y_dist

    # WIP!
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

    def crop_image(self, img):
        margin = int(CROP_IMAGE_SIZE / 2)

        # Get max brightness value of frame and its location
        _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)

        # Stop if point is not bright enough to be considered
        if brightest < MIN_BRIGHTNESS_FOR_PREDICTION:
            return None, brightest, (max_x, max_y)

        # Cut out region of interest around brightest point in image
        img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin]

        # If the point is close to the image border, the output image will be too small
        # TODO: Improve this later. Currently no need, as long as the camera FOV is larger than the projection area.
        # Problems only appear on the image border.
        if img_cropped.shape[0] != CROP_IMAGE_SIZE or img_cropped.shape[1] != CROP_IMAGE_SIZE:
            # img_cropped, brightest, (max_x, max_y) = self.crop_image_2(img)
            print('!#-#-#-#-#-#-#-#-#')
            print('Shape in crop 1:', img_cropped.shape, max_x, max_y)
            return None, brightest, (max_x, max_y)
            # time.sleep(20)

        # img_cropped_large = cv2.resize(img_cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('large', img_cropped_large)
        return img_cropped, brightest, (max_x, max_y)

    # TODO: CHECK this function. Currently not in use
    def crop_image_2(self, img):
        # print('using crop_image_2() function')
        margin = int(CROP_IMAGE_SIZE / 2)
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

        if left + CROP_IMAGE_SIZE >= img.shape[1]:
            # left -= (left + size - img.shape[1] - 1)
            left = img.shape[1] - CROP_IMAGE_SIZE - 1
        if top + CROP_IMAGE_SIZE >= img.shape[0]:
            # top -= (top + size - img.shape[0] - 1)
            top = img.shape[0] - CROP_IMAGE_SIZE - 1

        # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)
        img_cropped = img[top: top + CROP_IMAGE_SIZE, left: left + CROP_IMAGE_SIZE]

        print('Shape in crop 2:', img_cropped.shape, left + margin, top + margin)

        return img_cropped, np.max(img_cropped), (left + margin, top + margin)

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
            # print('small contour')
            # TODO: HOW TO HANDLE SMALL CONTOURS?
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
