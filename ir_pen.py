
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
            pen_event_roi, brightest, (x, y) = self.crop_image_old(frame)

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

        new_data = []

        for i, frame in enumerate(camera_frames):

            new_data.append([])

            # TODO: Check why we crash here sometimes
            try:
                # Extract all ROIs from the current frame
                rois_new, roi_coords_new, max_brightness_values = self.get_all_rois(frame)
            except Exception as e:
                print('ERROR in/after "self.get_all_rois(frame):"', e)
                rois_new, roi_coords_new, max_brightness_values = [], [], []

            for j, pen_event_roi in enumerate(rois_new):

                prediction, confidence = self.ir_pen_cnn.predict(pen_event_roi)
                new_transformed_coords = self.transform_coords_to_output_res(roi_coords_new[j][0], roi_coords_new[j][1], transform_matrices[i])

                # TODO: Rework subpixel coordinates calculation
                # (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)
                # subpixel_coords[i].append((x, y))

                new_data[i].append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'roi': pen_event_roi,
                    'max_brightness': max_brightness_values[j],
                    'transformed_coords': new_transformed_coords,
                    'subpixel_coords': new_transformed_coords,  # TODO: subpixel_coords should be used here!
                    'used': False  # Flag to check if the ROI has been used to generate a pen event
                })

        if DEBUG_MODE:
            self.preview_rois(new_data)
            self.preview_table_view(new_data)

        # TODO: Rework subpixel coordinates calculation
        # new_pen_events = self.generate_new_pen_events(subpixel_coords, predictions, brightness_values)
        new_pen_events = self.generate_new_pen_events_new_new(new_data)

        # if len(new_pen_events) > 0:
        #     print('New pen Events:', new_pen_events)

        # This function needs to be called even if there are no new pen events to update all existing events
        active_pen_events, stored_lines, pen_events_to_remove = self.pen_events_controller.merge_pen_events_new(new_pen_events)

        print('Active pen events', active_pen_events)

        # return [], [], [], [], []

        # TODO: REWORK RETURN. stored_lines not always needed
        return active_pen_events, stored_lines, new_pen_events, pen_events_to_remove

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

    def preview_table_view(self, new_data):

        zeros = np.zeros((2160, 3840, 3), 'uint8')

        for i in range(len(new_data)):
            for entry in new_data[i]:
                if i == 0:
                    cv2.circle(zeros, [int(entry['subpixel_coords'][0]), int(entry['subpixel_coords'][1])], 30, (255, 255, 255), -1)
                else:
                    cv2.circle(zeros, [int(entry['subpixel_coords'][0]), int(entry['subpixel_coords'][1])], 30, (60, 60, 60), -1)
                if entry['prediction'] == 'draw':
                    cv2.circle(zeros, [int(entry['subpixel_coords'][0]), int(entry['subpixel_coords'][1])], 20, (0, 255, 0), -1)
                else:
                    cv2.circle(zeros, [int(entry['subpixel_coords'][0]), int(entry['subpixel_coords'][1])], 20, (0, 0, 255), -1)

        cv2.imshow('Table preview', zeros)

    def preview_rois(self, new_data):

        preview_image_all_rois = None

        for i in range(len(new_data)):
            for entry in new_data[i]:
                roi_preview = entry['roi'].copy()
                roi_preview = cv2.resize(roi_preview, (960, 960), interpolation=cv2.INTER_AREA)
                roi_preview = self.add_label(roi_preview, entry['prediction'], 80)
                roi_preview = self.add_label(roi_preview, 'CAM ID: {}'.format(i), 130)
                roi_preview = self.add_label(roi_preview, 'MAX: {}'.format(int(entry['max_brightness'])), 180)

                if preview_image_all_rois is None:
                    preview_image_all_rois = roi_preview
                else:
                    preview_image_all_rois = cv2.hconcat([preview_image_all_rois, roi_preview])

        if preview_image_all_rois is not None:
            cv2.imshow('All ROIs', preview_image_all_rois)

    def add_label(self, frame, label, y_pos):
        return cv2.putText(
            img=frame,
            text=str(label),
            org=(90, y_pos),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(255),
            thickness=3
        )


    def generate_new_pen_events_new_new(self, new_data):

        new_pen_events = []

        # Every camera sees at least one point -> Check which points likely belong together
        if len(new_data[0]) > 0 and len(new_data[1]) > 0:
            point_distance_pairs_by_y_value = self.find_point_pairs_with_similar_y_value(new_data[0], new_data[1])
            # print(point_distance_pairs_by_y_value)

            for pair in point_distance_pairs_by_y_value:

                cam_0_index = pair[0]
                cam_1_index = pair[1]

                # If one of the points has already been used, we can continue
                if new_data[0][cam_0_index]['used'] or new_data[1][cam_1_index]['used']:
                    continue

                y_distance_between_points_abs = pair[2]
                x_distance_between_points = pair[3]

                prediction_0 = new_data[0][cam_0_index]['prediction']
                prediction_1 = new_data[1][cam_1_index]['prediction']

                # If both points have similar y values
                if y_distance_between_points_abs <= CROP_IMAGE_SIZE:
                    # L and R order impossible for a pair
                    if x_distance_between_points < -CROP_IMAGE_SIZE:
                        continue
                    # Two points are close on both x and y-axis
                    elif -CROP_IMAGE_SIZE <= x_distance_between_points <= CROP_IMAGE_SIZE:
                        if prediction_0 == 'draw' and prediction_1 == 'draw':
                            # New draw event from center point
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.generate_new_pen_event('draw', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True
                        elif prediction_0 == 'hover' and prediction_1 == 'hover':
                            # New hover event from center point
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.generate_new_pen_event('hover', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True
                        else:
                            # Maybe draw wins? and new event from center point?
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.generate_new_pen_event('draw', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True

                    # Points have similar y values but different x values
                    else:
                        if prediction_0 == 'draw' and prediction_1 == 'draw':
                            # Unlikely that two predictions are false -> Treat them as seperate events
                            continue
                        elif prediction_0 == 'hover' and prediction_1 == 'hover':
                            # New event from center point
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.generate_new_pen_event('hover', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True
                        else:
                            # difficult: Prediction of one point must be wrong or two separate events
                            continue
                else:
                    # Points do not have similar y-values. must be different events
                    continue

        # Deal with remaining points
        for i in range(len(new_data)):
            for entry in new_data[i]:
                if not entry['used']:
                    if entry['prediction'] == 'draw':
                        new_pen_events.append(self.generate_new_pen_event('draw',
                                                                          entry['subpixel_coords'][0],
                                                                          entry['subpixel_coords'][1]))
                        entry['used'] = True
                    elif entry['prediction'] == 'hover':
                        # Ignore for now
                        continue

        return new_pen_events

    # def generate_new_pen_events_new(self, new_data):
    #
    #     new_pen_events = []
    #
    #     draw_events = []
    #     hover_events = []
    #     used_indexes = []
    #
    #     for i in range(len(new_data)):
    #         draw_events.append([])
    #         hover_events.append([])
    #         used_indexes.append([])
    #
    #         for entry in new_data[i]:
    #             if entry['prediction'] == 'draw':
    #                 draw_events[i].append(entry)
    #             else:
    #                 hover_events[i].append(entry)
    #
    #     point_distance_pairs_by_y_value = self.find_point_pairs_with_similar_y_value(draw_events[0], draw_events[1])
    #     for pair in point_distance_pairs_by_y_value:
    #         cam_0_index = pair[0]
    #         cam_1_index = pair[1]
    #         distance_between_points = pair[2]
    #
    #         if distance_between_points <= CROP_IMAGE_SIZE:
    #             print('Found Draw connected draw event in both frames')
    #
    #             # We now have a point that is visible in both camera frames. Because those two frames might not be
    #             # perfectly aligned, we can use the center between both points here.
    #             (center_x, center_y) = self.get_center_point(draw_events[0][cam_0_index]['subpixel_coords'],
    #                                                          draw_events[1][cam_1_index]['subpixel_coords'])
    #
    #             new_pen_events.append(self.generate_new_pen_event('draw', center_x, center_y))
    #             used_indexes[0].append(cam_0_index)
    #             used_indexes[1].append(cam_1_index)
    #
    #     for i in range(len(draw_events)):
    #         for j in range(len(draw_events[i])):
    #             if j not in used_indexes[i]:
    #                 print('Found remaining draw event in single frame (Frame ID: {})'.format(i))
    #                 new_pen_events.append(self.generate_new_pen_event('draw',
    #                                                                   draw_events[i][j]['subpixel_coords'][0],
    #                                                                   draw_events[i][j]['subpixel_coords'][1]))
    #
    #     # TODO: Deal with remaining hover events
    #     for i in range(len(hover_events)):
    #         if len(hover_events[i]) > 0:
    #             print('Found {} remaining hover events for cam {}'.format(len(hover_events[i]), i))
    #
    #     return new_pen_events




    # WIP!
    # def generate_new_pen_events(self, coords, predictions, brightness_values):
    #
    #     PREVIEW_THIS = False
    #
    #     # TODO: REMOVE!
    #     # coords = [coords[0], coords[0]]
    #     # predictions = [predictions[0], predictions[0]]
    #
    #     new_pen_events = []
    #
    #     # if PREVIEW_THIS:
    #     #     zeros = np.zeros((2160, 3840, 3), 'uint8')
    #     #     for i, point_left_cam in enumerate(coords[0]):
    #     #         if predictions[0][i] == 'draw':
    #     #             cv2.circle(zeros, [int(point_left_cam[0]), int(point_left_cam[1])], 20, (0, 0, 255), -1)
    #     #         else:
    #     #             cv2.circle(zeros, [int(point_left_cam[0]), int(point_left_cam[1])], 20, (255, 0, 0), -1)
    #     #
    #     #     for i, point_right_cam in enumerate(coords[1]):
    #     #         if predictions[1][i] == 'draw':
    #     #             cv2.circle(zeros, [int(point_right_cam[0]), int(point_right_cam[1])], 20, (255, 0, 0), -1)
    #     #         else:
    #     #             cv2.circle(zeros, [int(point_right_cam[0]), int(point_right_cam[1])], 20, (0, 0, 255), -1)
    #
    #
    #
    #     # Shortcut for clear cases where one camera sees zero points -> No merge between both images needed
    #     # if len(coords[0]) == 0 and len(coords[1]) > 0 or len(coords[0]) > 0 and len(
    #     #         coords[1]) == 0:
    #     #     for i, point in enumerate(coords[0]):
    #     #         print('Single point at', point)
    #     #         new_pen_events.append(self.generate_new_pen_event(predictions[0][i], point[0], point[1]))
    #     #     for i, point in enumerate(coords[1]):
    #     #         print('Single point at', point)
    #     #         new_pen_events.append(self.generate_new_pen_event(predictions[1][i], point[0], point[1]))
    #
    #     if len(coords[0]) > 0 and len(coords[1]) > 0:
    #         # Returns a list with entries in the following format: [list0 index, list1 index, distance in px]
    #         # Currently we filter out all distances that are larger than max_distance. This will return only points
    #         # where we can assume that they are caused by the same pens
    #         shortest_distance_point_pairs = self.pen_events_controller.calculate_distances_between_all_points(coords[0], coords[1], max_distance=CROP_IMAGE_SIZE, as_objects=False)
    #
    #         point_distance_pairs_by_y_value = self.find_point_pairs_with_similar_y_value(coords[0], coords[1])
    #
    #         print(point_distance_pairs_by_y_value)
    #
    #         used_ids_cam_0 = []
    #         used_ids_cam_1 = []
    #
    #         # Iterate over all measured and prefiltered distance point pairs
    #         for entry in shortest_distance_point_pairs:
    #             cam_0_index = entry[0]
    #             cam_1_index = entry[1]
    #             # distance_between_points = entry[2]
    #
    #             if cam_0_index not in used_ids_cam_0 and cam_1_index not in used_ids_cam_1:
    #                 # y1 = coords[0][cam_0_index][1]
    #                 # y2 = coords[1][cam_1_index][1]
    #                 # if abs(y1 - y2) < MAX_DISTANCE_DRAW:
    #                 #     if predictions[0][cam_0_index] == 'draw' or predictions[1][cam_1_index] == 'draw':
    #
    #                         # if distance_between_points > CROP_IMAGE_SIZE:
    #                         #     continue
    #
    #                 # We now have a point that is visible in both camera frames. Because those two frames might not be
    #                 # perfectly aligned, we can use the center between both points here.
    #                 (center_x, center_y) = self.get_center_point(coords[0][cam_0_index], coords[1][cam_1_index])
    #
    #                 brightness_0 = brightness_values[0][cam_0_index]
    #                 brightness_1 = brightness_values[1][cam_1_index]
    #
    #                 # Compare brightness values and use prediction of the brighter point
    #                 if brightness_1 > brightness_0:
    #                     final_prediction = predictions[1][cam_1_index]
    #                 else:
    #                     final_prediction = predictions[0][cam_0_index]
    #
    #                 new_pen_events.append(self.generate_new_pen_event(final_prediction, center_x, center_y))
    #
    #                 used_ids_cam_0.append(cam_0_index)
    #                 used_ids_cam_1.append(cam_1_index)
    #
    #                 # if PREVIEW_THIS:
    #                 #     cv2.line(zeros,
    #                 #              [int(coords[0][cam_0_index][0]), int(coords[0][cam_0_index][1])],
    #                 #              [int(coords[1][cam_1_index][0]), int(coords[1][cam_1_index][1])], (255, 255, 255), 3)
    #
    #         # Now we did the easy ones.
    #         # TODO: Check the new points if they are draw/hover
    #
    #         # Now deal with the remaining points
    #         for i in range(len(coords[0])):
    #             if i not in used_ids_cam_0:
    #                 print('Point {} on cam 0 remains'.format(i))
    #                 new_pen_events.append(self.generate_new_pen_event(predictions[0][i], coords[0][i][0],
    #                                                                   coords[0][i][1]))
    #         for i in range(len(coords[1])):
    #             if i not in used_ids_cam_1:
    #                 print('Point {} on cam 1 remains'.format(i))
    #                 new_pen_events.append(self.generate_new_pen_event(predictions[1][i], coords[1][i][0],
    #                                                                   coords[1][i][1]))
    #
    #         # print(shortest_distance_point_pairs)
    #     # print(new_pen_events)
    #
    #     # for point_left_cam in subpixel_coords[0]:
    #     #     for point_right_cam in subpixel_coords[1]:
    #     #         dist_between_y = int(abs(point_left_cam[1] - point_right_cam[1]))
    #     #         distance_between_points = int(distance.euclidean(point_left_cam, point_right_cam))
    #     #
    #     #         print(distance_between_points, dist_between_y)
    #
    #     # if PREVIEW_THIS:
    #     #     cv2.imshow('preview', zeros)
    #
    #     return new_pen_events

    def find_point_pairs_with_similar_y_value(self, cam_0_coords, cam_1_coords):

        point_distance_pairs_by_y_value = []

        for i in range(len(cam_0_coords)):
            for j in range(len(cam_1_coords)):

                # Prevent out of range errors when lists have different lengths
                if i > len(cam_1_coords) -1 or j > len(cam_0_coords) - 1:
                    continue

                x1 = cam_0_coords[i]['subpixel_coords'][0]
                x2 = cam_1_coords[i]['subpixel_coords'][0]
                x_difference = x1 - x2

                y1 = cam_0_coords[i]['subpixel_coords'][1]
                y2 = cam_1_coords[j]['subpixel_coords'][1]
                y_difference_abs = abs(y1 - y2)
                point_distance_pairs_by_y_value.append([i, j, y_difference_abs, x_difference])

        # Sort list of lists by third element, in this case the y-distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        point_distance_pairs_by_y_value.sort(key=lambda x: x[2])

        return point_distance_pairs_by_y_value

    def generate_new_pen_event(self, prediction, x, y):
        if prediction == 'draw':
            new_ir_pen_event = PenEvent(x, y, PenState.DRAG)
            return new_ir_pen_event
        elif prediction == 'hover':
            new_ir_pen_event = PenEvent(x, y, PenState.HOVER)
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
    # def get_all_rois(self, img, size=CROP_IMAGE_SIZE):
    #
    #     rois = []
    #     roi_coords = []
    #     max_brightness_values = []
    #
    #     for i in range(10):
    #         margin = int(size / 2)
    #         _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)
    #
    #         # Dead pixel fix
    #         if max_x == 46 and max_y == 565:
    #             # TODO: Find solution for dead pixel
    #             continue
    #
    #         if brightest < MIN_BRIGHTNESS_FOR_PREDICTION:
    #             break
    #
    #         img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin].copy()
    #
    #         # print('Shape in crop 1:', img_cropped.shape, max_x, max_y)
    #
    #         if img_cropped.shape == (size, size):
    #             rois.append(img_cropped)
    #             roi_coords.append((max_x, max_y))
    #             max_brightness_values.append(int(brightest))
    #         else:
    #             print('TODO: WRONG SHAPE')
    #
    #         img.setflags(write=True)
    #         img[max_y - margin: max_y + margin, max_x - margin: max_x + margin] = 0
    #         img.setflags(write=False)
    #
    #     return rois, roi_coords, max_brightness_values

    def get_all_rois(self, image):

        # TODO: If you get too many ROIs, the exposure of the camera should be reduced
        # TODO: Maybe some sort of auto calibration could fix this

        rois_new = []
        roi_coords_new = []
        max_brightness_values = []

        margin = int(CROP_IMAGE_SIZE / 2)

        cutout = np.zeros((CROP_IMAGE_SIZE, CROP_IMAGE_SIZE), 'uint8')

        #print(image.shape)
        #print(cutout.shape)

        i = 0

        # TODO: Try to only copy the ROI and not the entire image. Does not work because I can't set the image flag to writeable.
        image_copy = image.copy()

        # while True:
        # try:
        while i < 10:

            # Get max brightness value of frame and its location
            _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image_copy)

            # Stop if point is not bright enough to be considered
            if brightest < MIN_BRIGHTNESS_FOR_PREDICTION:
                # print('Checks over, brightness to small')
                #print('Found a total of {} ROIs in frame'.format(len(rois_new)))
                #print(roi_coords_new)
                #print('')
                # return rois, roi_center_coordinates
                return rois_new, roi_coords_new, max_brightness_values

            #print('Brightest:', brightest)

            # Cut out region of interest around brightest point in image
            img_cropped = image[max_y - margin: max_y + margin, max_x - margin: max_x + margin]

            # print('Found new ROI candidate. Total:', len(rois))
            # print(img_cropped.shape)

            # print('Pixel value:', image_copy[max_y, max_x])

            # print('Iteration:', i)
            i += 1

            # If the point is close to the image border, the output image will be too small
            # TODO: Improve this later. Currently no need, as long as the camera FOV is larger than the projection area.
            # Problems only appear on the image border.
            if img_cropped.shape[0] == CROP_IMAGE_SIZE and img_cropped.shape[1] == CROP_IMAGE_SIZE:

                # image.setflags(write=1)
                # Set all pixels in ROI to black
                image_copy[max_y - margin: max_y + margin, max_x - margin: max_x + margin] = cutout
                # image.setflags(write=0)

                rois_new.append(img_cropped)
                roi_coords_new.append((max_x, max_y))
                max_brightness_values.append(brightest)


                if len(rois_new) > 3:
                    print('WHY SO MANY?')

            else:
                # print('ROI shape too small')

                # Set just this pixel to black
                # TODO: Needs improvement, set also the surrounding area to black
                image_copy[max_y, max_x] = np.zeros((1, 1, 1), 'uint8')

    def crop_image_old(self, image):
        margin = int(CROP_IMAGE_SIZE / 2)

        # Get max brightness value of frame and its location
        _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image)

        # Stop if point is not bright enough to be considered
        if brightest < MIN_BRIGHTNESS_FOR_PREDICTION:
            return None, brightest, (max_x, max_y)

        # Cut out region of interest around brightest point in image
        img_cropped = image[max_y - margin: max_y + margin, max_x - margin: max_x + margin]

        # If the point is close to the image border, the output image will be too small
        # TODO: Improve this later. Currently no need, as long as the camera FOV is larger than the projection area.
        # Problems only appear on the image border.
        if img_cropped.shape[0] != CROP_IMAGE_SIZE or img_cropped.shape[1] != CROP_IMAGE_SIZE:
            # img_cropped, brightest, (max_x, max_y) = self.crop_image_2(image)
            print('!#-#-#-#-#-#-#-#-#')
            print('Shape in crop 1:', img_cropped.shape, max_x, max_y)
            return None, brightest, (max_x, max_y)
            # time.sleep(20)

        # img_cropped_large = cv2.resize(img_cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('large', img_cropped_large)
        return img_cropped, brightest, (max_x, max_y)

    # TODO: CHECK this function. Currently not in use
    def crop_image_2_old(self, img):
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
            # left -= (left + size - image.shape[1] - 1)
            left = img.shape[1] - CROP_IMAGE_SIZE - 1
        if top + CROP_IMAGE_SIZE >= img.shape[0]:
            # top -= (top + size - image.shape[0] - 1)
            top = img.shape[0] - CROP_IMAGE_SIZE - 1

        # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image)
        img_cropped = img[top: top + CROP_IMAGE_SIZE, left: left + CROP_IMAGE_SIZE]

        print('Shape in crop 2:', img_cropped.shape, left + margin, top + margin)

        return img_cropped, np.max(img_cropped), (left + margin, top + margin)

    # TODO: Fix possible offset
    def find_pen_position_subpixel_crop(self, roi, center_original):
        w = roi.shape[0]
        h = roi.shape[1]
        # print('1', ir_image.shape)
        # center_original = (coords_original[0] + w/2, coords_original[1] + h/2)

        new_w = int(w * self.factor_width)
        new_h = int(h * self.factor_height)
        top_left_scaled = (center_original[0] * self.factor_width - new_w / 2,
                           center_original[1] * self.factor_height - new_h / 2)

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
