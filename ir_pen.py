
from enum import Enum
import time
import numpy as np
from tensorflow import keras
from tflite import LiteModel
import datetime
from scipy.spatial import distance

from cv2 import cv2

# TODO: Relative Path
MODEL_PATH = 'evaluation/104_2022-06-10'

CROP_IMAGE_SIZE = 48

# Simple Smoothing
SMOOTHING_FACTOR = 0.99  # Value between 0 and 1, depending on if the old or the new value should count more.


# Amount of time a point can be missing until the event "on click/drag stop" will be fired
TIME_POINT_MISSING_THRESHOLD_MS = 22

# Point needs to appear and disappear within this timeframe in ms to count as a click (vs. a drag event)
CLICK_THRESH_MS = 10

# Hover will be selected over Draw if Hover Event is within the last X event states
KERNEL_SIZE_HOVER_WINS = 3


DEBUG_MODE = True

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

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
        self.state_history = []

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
        #print(np.std(img_cropped), flush=True)
        #print(min_radius, flush=True)

        if DEBUG_MODE:
            preview = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

        MIN_BRIGHTNESS = 100

        # TODO: for loop here to iterate over all detected bright spots in the image
        if brightest > MIN_BRIGHTNESS and img_cropped.shape == (CROP_IMAGE_SIZE, CROP_IMAGE_SIZE):
            prediction, confidence = self.predict(img_cropped)
            #print(confidence, flush=True)

            min_radius, coords = self.find_pen_position(ir_frame)
            #y_left, y_right = self.find_pen_orientation(ir_frame)
            color = (0, 0, 0)

            (x, y) = self.convert_coordinate_to_target_resolution(coords[0], coords[1], ir_frame.shape[1], ir_frame.shape[0], WINDOW_WIDTH, WINDOW_HEIGHT)
            print('old', (x, y))
            (x, y) = self.find_pen_position_subpixel(ir_frame)
            print('new', (x, y))

            if prediction == 'draw':
                # print('Status: Touch')
                new_ir_pen_event = PenEvent(x, y)
                new_ir_pen_event.state = State.DRAG
                new_pen_events.append(new_ir_pen_event)

                color = (0, 255, 0)
            elif prediction == 'hover':
                # print('Status: Hover')
                new_ir_pen_event = PenEvent(x, y)
                new_ir_pen_event.state = State.HOVER
                new_pen_events.append(new_ir_pen_event)
                color = (0, 0, 255)
                #if y_left > -1 and y_right > -1:
                #    preview = cv2.line(preview, (ir_frame.shape[1]-1,y_right),(0,y_left),(0,255,0),1)
            else:
                print('Unknown state')

            if DEBUG_MODE:
                #preview = cv2.circle(preview, coords, 10, color, -1)
                preview = cv2.rectangle(preview, (coords[0] - 24, coords[1] - 24), (coords[0] + 24, coords[1] + 24), color, 1)

        # if DEBUG_MODE:
        #     cv2.imshow('preview', preview)
        #     cv2.waitKey(1)

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

    def find_pen_position_subpixel(self, ir_image):
        _, thresh = cv2.threshold(ir_image, np.max(ir_image) - 80, 255, cv2.THRESH_BINARY)

        thresh_large = cv2.resize(thresh, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
        # cv2.imshow('thresh large', thresh_large)
        contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        min_radius = ir_image.shape[0]
        smallest_contour = contours[0]

        print(len(contours))
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                smallest_contour = contour

        M = cv2.moments(smallest_contour)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])

        cY = int(M["m01"] / M["m00"])

        position = (cX, cY)

        return position

        # left_ir_image_large = cv2.resize(left_ir_image.copy(), (3840, 2160), interpolation=cv2.INTER_AREA)
        # cv2.circle(left_ir_image_large, position, 1, (0, 0, 0))
        # cv2.imshow('large', left_ir_image_large)

    # Finds the exact center
    def find_pen_position(self, img):
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

    def find_pen_orientation(self, image):
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
        # Get current timestamp
        now = round(time.time() * 1000)

        for new_pen_event in new_pen_events:
            new_pen_event.state_history = [new_pen_event.state]

        # if DEBUG_MODE:
        #     for final_pen_event in self.active_pen_events:
        #         # print(final_pen_event)
        #         try:
        #             print(final_pen_event.id, max(final_pen_event.state_history[-5:], key=final_pen_event.state_history[-5:].count))
        #             print(final_pen_event.state_history[-5:])
        #         except:
        #             print('Oh')

        # Iterate over copy of list
        # If a final_pen_event has been declared a "Click Event" in the last frame, this event is now over, and we can delete it.
        for active_pen_event in self.active_pen_events[:]:
            if active_pen_event.state == State.CLICK:
                self.process_click_events(active_pen_event)

        # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
        shortest_distance_point_pairs = self.calculate_distances_between_all_points(new_pen_events)

        for entry in shortest_distance_point_pairs:
            last_pen_event = self.active_pen_events[entry[0]]
            new_pen_event = new_pen_events[entry[1]]

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

            if State.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:
                # print('Hover wins')
                new_pen_event.state = State.HOVER
            else:
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
            new_pen_event.x = int(SMOOTHING_FACTOR * (new_pen_event.x - last_pen_event.x) + last_pen_event.x)
            new_pen_event.y = int(SMOOTHING_FACTOR * (new_pen_event.y - last_pen_event.y) + last_pen_event.y)

            # Set the ID of the last_pen_event back to -1 so that it is ignored in all future checks
            # We later want to only look at the remaining last_pen_event that did not have a corresponding new_pen_event
            last_pen_event.id = -1

        for new_pen_event in new_pen_events:
            new_pen_event.missing = False
            new_pen_event.last_seen_timestamp = now

        # Check all active_pen_events that do not have a match found after comparison with the new_pen_events
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
