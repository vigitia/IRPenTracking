
import sys
import time
import datetime
import numpy as np
import skimage
import cv2

from TipTrack.pen_events.pen_state import PenState
from TipTrack.pen_events.pen_event import PenEvent
from TipTrack.pen_events.pen_events_controller import PenEventsController
from TipTrack.pen_events.ir_pen_cnn import IRPenCNN

from TipTrack.utility.roi_extractor import ROIExtractor


CROP_IMAGE_SIZE = 48  # Currently 48x48 Pixel. Relevant for the CNN

USE_MAX_DISTANCE_DRAW = False
MAX_DISTANCE_DRAW = 500  # Maximum allowed pixel distance between two points to be considered the same line ID

# LATENCY_MEASURING_MODE = False

# Width and height of the output window/screen -> Target resolution
# ToDo: Store this data somewhere else
OUTPUT_WINDOW_WIDTH = 3840
OUTPUT_WINDOW_HEIGHT = 2160

# Width and height of the received frames
INPUT_FRAME_WIDTH = 1920  # 848
INPUT_FRAME_HEIGHT = 1200  # 480

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

    factor_width = OUTPUT_WINDOW_WIDTH / INPUT_FRAME_WIDTH
    factor_height = OUTPUT_WINDOW_HEIGHT / INPUT_FRAME_HEIGHT

    def __init__(self, debug_mode=DEBUG_MODE):
        global DEBUG_MODE
        DEBUG_MODE = debug_mode

        if DEBUG_MODE:
            cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.ir_pen_cnn = IRPenCNN()
        self.pen_events_controller = PenEventsController()
        self.roi_extractor = ROIExtractor(CROP_IMAGE_SIZE)

    def get_ir_pen_events(self, camera_frames, transform_matrices):

        new_data = self.get_new_pen_data(camera_frames, transform_matrices)

        if DEBUG_MODE:
            self.__debug_mode_preview_rois(new_data)
            self.__debug_mode_preview_table_view(new_data)

        # TODO: Rework subpixel coordinates calculation
        new_pen_events = self.generate_new_pen_events(new_data)

        if len(new_pen_events) > 2:
            print('[IRPen]: TOO MANY NEW PEN EVENTS!')

        # This function needs to be called even if there are no new pen events to update all existing events
        active_pen_events, stored_lines, pen_events_to_remove = self.pen_events_controller.merge_pen_events_new(new_pen_events)

        active_pen_events.sort(key=lambda x: x.id, reverse=False)

        # TODO: REWORK RETURN. stored_lines not always needed
        return active_pen_events, stored_lines, pen_events_to_remove

    # @ timeit('get_new_pen_data')
    def get_new_pen_data(self, camera_frames, transform_matrices):
        new_data = []

        for i, frame in enumerate(camera_frames):

            new_data.append([])

            rois_new, roi_coords_new, max_brightness_values = self.roi_extractor.get_all_rois(frame)

            for j, pen_event_roi in enumerate(rois_new):
                prediction, confidence = self.ir_pen_cnn.get_prediction(pen_event_roi)
                new_transformed_coords = self.transform_coords_to_output_res(roi_coords_new[j][0], roi_coords_new[j][1],
                                                                             transform_matrices[i])

                # TODO: Rework subpixel coordinates calculation
                # (x, y), radius = self.find_pen_position_subpixel_crop(pen_event_roi, transformed_coords)
                # subpixel_coords[i].append((x, y))

                new_event = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'roi': pen_event_roi,
                    'max_brightness': max_brightness_values[j],
                    'transformed_coords': new_transformed_coords,
                    'subpixel_coords': new_transformed_coords,  # TODO: subpixel_coords should be used here!
                    'used': False  # Flag to check if the ROI has been used to generate a pen event
                }

                new_data[i].append(new_event)

        return new_data

    # By using the transform matrix handed over by the camera, the given coordinate will be transformed from camera
    # space to projector space
    def transform_coords_to_output_res(self, x, y, transform_matrix):

        try:
            coords = np.array([x, y, 1])

            transformed_coords = transform_matrix.dot(coords)

            # Normalize coordinates by dividing by z
            transformed_coords = (transformed_coords[0] / transformed_coords[2],
                                  transformed_coords[1] / transformed_coords[2])

            # Coordinates are now aligned with the projection area but still need to be upscaled to the output
            # resolution
            transformed_coords = (transformed_coords[0] * self.factor_width, transformed_coords[1] * self.factor_height)

            return transformed_coords
        except Exception as e:
            print('[IRPen]: Error in transform_coords_to_output_res(). Maybe the transform_matrix is malformed?')
            print(e)
            sys.exit(1)

    def __debug_mode_preview_table_view(self, new_data):

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

    def __debug_mode_preview_rois(self, new_data):
        """ Debug Mode Preview ROIs

            Requires this method to be called from the main thread. Otherwise, cv2.imshow won't work.


        """

        preview_image_all_rois = None

        preview_roi_frame = np.zeros((OUTPUT_WINDOW_HEIGHT, OUTPUT_WINDOW_WIDTH), 'uint8')

        for i in range(len(new_data)):
            for entry in new_data[i]:

                if i == 0:
                    print(preview_roi_frame.shape, entry['roi'].shape)
                    roi = entry['roi']
                    w = roi.shape[1]
                    h = roi.shape[0]
                    x = int(entry['subpixel_coords'][0])
                    y = int(entry['subpixel_coords'][1])
                    print(int(y-h/2), int(y+h/2), int(x-w/2), int(x+w/2))

                    try:
                        preview_roi_frame[int(y-h/2): int(y+h/2), int(x-w/2): int(x+w/2)] = entry['roi']
                    except Exception as e:
                        print(e)

                roi_preview = entry['roi'].copy()
                roi_preview = cv2.resize(roi_preview, (960, 960), interpolation=cv2.INTER_AREA)
                roi_preview = self.__add_label(roi_preview, entry['prediction'], 80)
                roi_preview = self.__add_label(roi_preview, 'CAM ID: {}'.format(i), 130)
                roi_preview = self.__add_label(roi_preview, 'MAX: {}'.format(int(entry['max_brightness'])), 180)
                roi_preview = self.__add_label(roi_preview, 'POS: {}, {}'.format(int(entry['subpixel_coords'][0]),
                                                                                 int(entry['subpixel_coords'][1])), 230)

                if preview_image_all_rois is None:
                    preview_image_all_rois = roi_preview
                else:
                    preview_image_all_rois = cv2.hconcat([preview_image_all_rois, roi_preview])

        cv2.imshow('Overlay', preview_roi_frame)

        if preview_image_all_rois is not None:
            cv2.imshow('All ROIs', preview_image_all_rois)

    @staticmethod
    def __add_label(frame, label, y_pos):
        return cv2.putText(
            img=frame,
            text=str(label),
            org=(90, y_pos),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(255),
            thickness=3
        )

    def generate_new_pen_events(self, new_data):

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

                n = 2

                # If both points have similar y values
                if y_distance_between_points_abs <= n * CROP_IMAGE_SIZE:
                    # L and R order impossible for a pair
                    if x_distance_between_points < - n * CROP_IMAGE_SIZE:
                        continue
                    # Two points are close on both x and y-axis
                    elif -n*CROP_IMAGE_SIZE <= x_distance_between_points <= n*CROP_IMAGE_SIZE:
                        if prediction_0 == 'draw' and prediction_1 == 'draw':
                            # New draw event from center point
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.__generate_new_pen_event('draw', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True
                        elif prediction_0 == 'hover' and prediction_1 == 'hover':
                            # New hover event from center point
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.__generate_new_pen_event('hover', center_x, center_y))
                            new_data[0][cam_0_index]['used'] = True
                            new_data[1][cam_1_index]['used'] = True
                        else:
                            # Maybe draw wins? and new event from center point?
                            (center_x, center_y) = self.get_center_point(new_data[0][cam_0_index]['subpixel_coords'],
                                                                         new_data[1][cam_1_index]['subpixel_coords'])

                            new_pen_events.append(self.__generate_new_pen_event('draw', center_x, center_y))
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

                            new_pen_events.append(self.__generate_new_pen_event('hover', center_x, center_y))
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
                        new_pen_events.append(self.__generate_new_pen_event('draw',
                                                                            entry['subpixel_coords'][0],
                                                                            entry['subpixel_coords'][1]))
                        entry['used'] = True
                    elif entry['prediction'] == 'hover':
                        # Ignore for now
                        continue

        return new_pen_events

    def find_point_pairs_with_similar_y_value(self, cam_0_coords, cam_1_coords):

        point_distance_pairs_by_y_value = []

        for i in range(len(cam_0_coords)):
            for j in range(len(cam_1_coords)):

                # Prevent out of range errors when lists have different lengths
                if i > len(cam_1_coords) - 1 or j > len(cam_0_coords) - 1:
                    continue

                x1 = cam_0_coords[i]['subpixel_coords'][0]
                x2 = cam_1_coords[j]['subpixel_coords'][0]
                x_difference = x1 - x2

                y1 = cam_0_coords[i]['subpixel_coords'][1]
                y2 = cam_1_coords[j]['subpixel_coords'][1]
                y_difference_abs = abs(y1 - y2)
                point_distance_pairs_by_y_value.append([i, j, y_difference_abs, x_difference])

        # Sort list of lists by third element, in this case the y-distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        point_distance_pairs_by_y_value.sort(key=lambda x: x[2])

        return point_distance_pairs_by_y_value

    def __generate_new_pen_event(self, prediction, x, y):
        """ Generate a single new Pen Event

        For a new Pen Event, we need the Type of the Pen Event (PenState), e.g. draw or hover and the x and y position.
        """
        if prediction == 'draw':
            new_ir_pen_event = PenEvent(x, y, PenState.DRAG)
            return new_ir_pen_event
        elif prediction == 'hover':
            new_ir_pen_event = PenEvent(x, y, PenState.HOVER)
            return new_ir_pen_event
        else:
            raise Exception('Error: Unknown/Unexpected Pen Event state')
            # sys.exit(1)

    # Helper Function: Calculate the center point between two given points
    def get_center_point(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2

        x_dist = abs(x1 - x2) / 2
        y_dist = abs(y1 - y2) / 2
        return min(x1, x2) + x_dist, min(y1, y2) + y_dist

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