
from enum import Enum
import time
import numpy as np
from tensorflow import keras
from tflite import LiteModel
import datetime
from scipy.spatial import distance

from cv2 import cv2

# TODO: Relative Path
MODEL_PATH = '/home/vigitia/Desktop/GitHub/IRPenTracking/evaluation/hover_predictor_binary_7'

CROP_IMAGE_SIZE = 48

# Simple Smoothing
SMOOTHING_FACTOR = 0.6  # Value between 0 and 1, depending on if the old or the new value should count more.


# Amount of time a point can be missing until the event "on click/drag stop" will be fired
TIME_POINT_MISSING_THRESHOLD_MS = 10

# Point needs to appear and disappear within this timeframe in ms to count as a click (vs. a drag event)
CLICK_THRESH_MS = 10


DEBUG_MODE = False

# TODO: Change these states
STATES = ['draw', 'hover', 'undefined']


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
        self.history = []

        self.alive = True

    def get_coordinates(self):
        return tuple([self.x, self.y])

    def __repr__(self):
        return 'Pen Event {} at ({}, {}). Type: {}. Num Points: {}'.format(str(self.id), str(self.x), str(self.y),
                                                                           self.state, len(self.history))


class IRPen:

    active_pen_events = []
    highest_id = 1
    stored_lines = []
    pen_events_to_remove = []  # Points that got deleted from active_points in the current frame
    double_click_candidates = []

    def __init__(self):
        keras.backend.clear_session()

        model = keras.models.load_model(MODEL_PATH)
        self.keras_lite_model = LiteModel.from_keras_model(model)

    def get_ir_pen_events(self, ir_frame):
        new_pen_events = []

        self.new_lines = []
        self.pen_events_to_remove = []

        # TODO: Get here all spots and not just one
        img_cropped, brightest, (x, y) = self.crop_image(ir_frame)

        WINDOW_WIDTH = 3840
        WINDOW_HEIGHT = 2160
        (x, y) = self.convert_coordinate_to_target_resolution(x, y, ir_frame.shape[1], ir_frame.shape[0], WINDOW_WIDTH, WINDOW_HEIGHT)

        # TODO: for loop here to iterate over all detected bright spots in the image
        if brightest > 100 and img_cropped.shape == (CROP_IMAGE_SIZE, CROP_IMAGE_SIZE):
            prediction, confidence = self.predict(img_cropped)

            if prediction == 'draw':
                print('Status: Touch')
                new_ir_pen_event = PenEvent(x, y)
                new_pen_events.append(new_ir_pen_event)

                if DEBUG_MODE:
                    cv2.imshow('spots', ir_frame)
            elif prediction == 'hover':
                print('Status: Hover')
            else:
                print('Unknown state')

        self.active_pen_events = self.merge_pen_events(new_pen_events)

        return self.active_pen_events, self.stored_lines, self.new_lines, self.pen_events_to_remove

    def convert_coordinate_to_target_resolution(self, x, y, current_res_x, current_res_y, target_x, target_y):
        x_new = int((x / current_res_x) * target_x)
        y_new = int((y / current_res_y) * target_y)

        return x_new, y_new

    def crop_image(self, img, size=CROP_IMAGE_SIZE):
        margin = int(size / 2)
        _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)
        img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin]
        return img_cropped, brightest, (max_x, max_y)

    def predict(self, img):
        img = img.astype('float32') / 255
        img = img.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 1)
        prediction = self.keras_lite_model.predict(img)
        if not prediction.any():
            return STATES[-1], 0
        state = STATES[np.argmax(prediction)]
        confidence = np.max(prediction)
        return state, confidence

    def merge_pen_events(self, new_pen_events):
        # Get current timestamp
        now = round(time.time() * 1000)

        if DEBUG_MODE:
            for final_pen_event in self.active_pen_events:
                print(final_pen_event)

        # Iterate over copy of list
        # If a final_pen_event has been declared a "Click Event" in the last frame, this event is now over, and we can delete it.
        for active_pen_event in self.active_pen_events[:]:
            if active_pen_event.state == State.CLICK:
                self.process_click_events(active_pen_event)

        # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
        shortest_distance_point_pairs = self.calculate_distances_between_all_points(new_pen_events)

        for entry in shortest_distance_point_pairs:
            active_pen_event = self.active_pen_events[entry[0]]
            new_pen_event = new_pen_events[entry[1]]

            if active_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
                print('HOVER EVENT turned into TOUCH EVENT')

            # Move ID and other important information from the active touch final_pen_event into the new
            # touch final_pen_event
            new_pen_event.id = active_pen_event.id
            new_pen_event.first_appearance = active_pen_event.first_appearance
            new_pen_event.state = active_pen_event.state
            new_pen_event.history = active_pen_event.history
            new_pen_event.x = int(SMOOTHING_FACTOR * (new_pen_event.x - active_pen_event.x) + active_pen_event.x)
            new_pen_event.y = int(SMOOTHING_FACTOR * (new_pen_event.y - active_pen_event.y) + active_pen_event.y)

            # Set the ID of the active_pen_event back to -1 so that it is ignored in all future checks
            # We later want to only look at the remaining active_pen_events that did not have a corresponding new_pen_event
            active_pen_event.id = -1

        for new_pen_event in new_pen_events:
            new_pen_event.missing = False
            new_pen_event.last_seen_timestamp = now

        # Check all active_pen_events that do not have a match found after comparison with the new_pen_events
        for active_pen_event in self.active_pen_events:
            # Skip all active_pen_events with ID -1. For those we already have found a matc.h
            if active_pen_event.id == -1:
                continue

            time_since_last_seen = now - active_pen_event.last_seen_timestamp

            if not active_pen_event.missing or time_since_last_seen < TIME_POINT_MISSING_THRESHOLD_MS:
                if not active_pen_event.missing:
                    active_pen_event.last_seen_timestamp = now

                active_pen_event.missing = True
                new_pen_events.append(active_pen_event)

            else:
                if active_pen_event.state == State.NEW:
                    # We detected a click event but we do not remove it yet because it also could be a double click.
                    # We will check this the next time this function is called.
                    active_pen_event.state = State.CLICK
                    new_pen_events.append(active_pen_event)
                elif active_pen_event.state == State.DRAG:
                    # End of a drag event
                    print('DRAG END')
                    self.pen_events_to_remove.append(active_pen_event)
                    self.stored_lines.append(np.array(active_pen_event.history))
                    self.new_lines.append(active_pen_event.history)
                elif active_pen_event.state == State.HOVER:
                    # End of a Hover event
                    print('HOVER EVENT END')
                    self.pen_events_to_remove.append(active_pen_event)

        final_pen_events = self.assign_new_ids(new_pen_events)

        for final_pen_event in final_pen_events:
            # Add current position to the history list
            final_pen_event.history.append((final_pen_event.x, final_pen_event.y))

            time_since_first_appearance = now - final_pen_event.first_appearance
            if final_pen_event.state != State.CLICK and final_pen_event.state != State.DOUBLE_CLICK and time_since_first_appearance > CLICK_THRESH_MS:
                if final_pen_event.state == State.NEW:
                    # Start of a drag event
                    print('DRAG START')
                    final_pen_event.state = State.DRAG
                elif final_pen_event.state == State.HOVER:
                    print('DETECTED Hover EVENT!')

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

    def calculate_distances_between_all_points(self, new_pen_events):
        distances = []

        for i in range(len(self.active_pen_events)):
            for j in range(len(new_pen_events)):
                distance_between_points = distance.euclidean(self.active_pen_events[i].get_coordinates(),
                                                             new_pen_events[j].get_coordinates())
                distances.append([i, j, distance_between_points])

        # Sort list of lists by third element, in this case the distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        distances.sort(key=lambda x: x[2])

        return distances