
import time
import numpy as np
from scipy.spatial import distance

from TipTrack.pen_events.pen_state import PenState


# Hover will be selected over Draw if Hover Event is within the last X event states.
# Enable this if too many unwanted short lines appear while drawing
# HOVER_WINS = False
# NUM_CHECK_LAST_EVENT_STATES = 3  # Check the last X Pen Events if they contain a hover event

NUM_HOVER_EVENTS_TO_END_LINE = 5  # We expect at least X Hover Events in a row to end that pen event

# Amount of time a pen can be missing until the pen event will be ended
TIME_POINT_MISSING_THRESHOLD_MS = 60  # 15

MAX_DISTANCE_FOR_MERGE = 500  # 200  # Maximum Distance between two points for them to be able to merge

# Simple Smoothing of the output points. 1 -> No smoothing; 0.5 -> Calculate a new point and use 50% of the previous
# point's and 50% of the new point's location.
SMOOTHING_FACTOR = 0.2 #  # 0.2  # Value between 0 and 1, depending on if the old or the new value should count more.

DEBUG_MODE = False


class PenEventsController:

    active_pen_events = []

    # double_click_candidates = []

    # If a pen event is over, it will be deleted. But its points will be stored here to keep track of all drawn lines.
    stored_lines = []

    highest_id = 0  # Assign each new pen event a new id. This variable keeps track of the highest number.

    def merge_pen_events_new(self, new_pen_events):
        pen_events_to_remove = []  # Points that got deleted from active_points in the current frame

        now = time.time_ns() // 1_000_000  # round(time.time() * 1000)  # Get current timestamp

        # No new events. Check active events if missing for too long
        if len(new_pen_events) == 0 and len(self.active_pen_events) > 0:

            remaining_events = []

            # Remove events that are missing for too long
            for active_pen_event in self.active_pen_events:
                # print('Remaining active event but now new event to merge with')
                time_since_last_seen = now - active_pen_event.last_seen_timestamp

                if time_since_last_seen > TIME_POINT_MISSING_THRESHOLD_MS:
                    print('Delete PenEvent {} ({} points): Inactive with last state {} while no new pen events present'.format(active_pen_event.id, len(active_pen_event.history), active_pen_event.state))
                    if len(active_pen_event.history) > 0:
                        # self.stored_lines.append(np.array(active_pen_event.history))
                        self.stored_lines.append({active_pen_event.id: active_pen_event.history})

                    pen_events_to_remove.append(active_pen_event)
                else:
                    remaining_events.append(active_pen_event)

            self.active_pen_events = remaining_events
            return self.active_pen_events, self.stored_lines, pen_events_to_remove

        final_pen_events = []

        # Merge new and old
        # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
        shortest_distance_point_pairs = self.__calculate_distances_between_all_points(self.active_pen_events, new_pen_events)

        # This is not the event ID but the index pos in the list
        merged_ids_active_pen_events = []
        merged_ids_new_pen_events = []

        for entry in shortest_distance_point_pairs:

            # Continue if event has already been merged
            if entry[0] in merged_ids_active_pen_events or entry[1] in merged_ids_new_pen_events:
                continue

            active_pen_event = self.active_pen_events[entry[0]]
            new_pen_event = new_pen_events[entry[1]]
            distance_between_points = entry[2]

            if distance_between_points <= MAX_DISTANCE_FOR_MERGE:

                # Merge events
                merged_event = self.__merge_events(active_pen_event, new_pen_event, now,
                                                   self.active_pen_events, new_pen_events)
                final_pen_events.append(merged_event)

                merged_ids_active_pen_events.append(entry[0])
                merged_ids_new_pen_events.append(entry[1])

        # Iterate over all remaining events in both active_pen_events and new_pen_events
        # Add remaining new_pen_events directly to final_pen_events
        # Check remaining active_pen_events if they are still relevant
        for i, new_pen_event in enumerate(new_pen_events):
            if i not in merged_ids_new_pen_events:
                final_pen_events.append(new_pen_event)

        for i, active_pen_event in enumerate(self.active_pen_events):
            if i not in merged_ids_active_pen_events:
                time_since_last_seen = now - active_pen_event.last_seen_timestamp

                # Mögliches Ende einer Linie
                if time_since_last_seen > TIME_POINT_MISSING_THRESHOLD_MS:
                    print('Delete PenEvent {} ({} points): Inactive with last state {}'.format(
                        active_pen_event.id, len(active_pen_event.history), active_pen_event.true_state))
                    if len(active_pen_event.history) > 0:
                        # self.stored_lines.append(np.array(active_pen_event.history))
                        self.stored_lines.append({active_pen_event.id: active_pen_event.history})

                    pen_events_to_remove.append(active_pen_event)
                else:
                    final_pen_events.append(active_pen_event)

        final_pen_events = self.__assign_new_ids(final_pen_events)

        # TODO: Improve this removal of elements from list
        final_final_pen_events = []
        for final_pen_event in final_pen_events:

            # Advanced filter of unexpected events: Single hover to drag
            if len(final_pen_event.state_history) >= 2 and final_pen_event.state == PenState.HOVER and final_pen_event.state_history[-2] == PenState.DRAG:
                # print('Single hover to drag')
                final_pen_event.state = PenState.DRAG

            # Advanced filter of unexpected events: Single drag to hover
            if len(final_pen_event.state_history) >= 2 and final_pen_event.state == PenState.DRAG and \
                    final_pen_event.state_history[-2] == PenState.HOVER:
                # print('Single drag to hover')
                final_pen_event.state = PenState.HOVER

            # Check if there are too many hover events. End event if this is the case
            if len(final_pen_event.state_history) >= NUM_HOVER_EVENTS_TO_END_LINE:

                # There needs to be at least one Draw event to delete a line (except through inactivity)
                if PenState.DRAG in final_pen_event.state_history:
                    if final_pen_event.state_history[-NUM_HOVER_EVENTS_TO_END_LINE:].count(PenState.HOVER) == NUM_HOVER_EVENTS_TO_END_LINE:

                        final_pen_event.history.append((final_pen_event.x, final_pen_event.y))
                        self.stored_lines.append({final_pen_event.id: final_pen_event.history})
                        pen_events_to_remove.append(final_pen_event)

                        print('Delete PenEvent {} ({} points): DRAW ended'.format(final_pen_event.id, len(final_pen_event.history)))

                        continue

                    # if final_pen_event.state == PenState.HOVER:
                    #     if final_pen_event.state_history.count(PenState.DRAG) > NUM_HOVER_EVENTS_TO_END_LINE:
                    #         print('Hover --> DRAG')
                    #         final_pen_event.state = PenState.DRAG

            final_final_pen_events.append(final_pen_event)

        self.active_pen_events = final_final_pen_events

        return self.active_pen_events, self.stored_lines, pen_events_to_remove

    # Combine a PenEvent from the previous frame with a new PenEvent from the current frame
    def __merge_events(self, active_pen_event, new_pen_event, current_timestamp, active_pen_events, new_pen_events):
        if (distance.euclidean(new_pen_event.get_coordinates(), active_pen_event.get_coordinates())) > 200:
            # print('Dist')
            pass

        new_pen_event.id = active_pen_event.id  # Use existing ID
        new_pen_event.last_seen_timestamp = current_timestamp
        new_pen_event.first_appearance = active_pen_event.first_appearance

        new_pen_event.history = active_pen_event.history  # Transfer over history
        new_pen_event.history.append((active_pen_event.x, active_pen_event.y))  # Add the last point from the old event

        new_pen_event.state_history = active_pen_event.state_history  # Transfer over state history
        new_pen_event.state_history.append(active_pen_event.true_state)  # Add last state from old PenEvent

        # Apply smoothing to the points by taking their previous positions into account
        new_pen_event.x = SMOOTHING_FACTOR * (new_pen_event.x - active_pen_event.x) + active_pen_event.x
        new_pen_event.y = SMOOTHING_FACTOR * (new_pen_event.y - active_pen_event.y) + active_pen_event.y

        return new_pen_event

    def __assign_new_ids(self, new_pen_events):
        final_pen_events = []

        for new_pen_event in new_pen_events:
            if new_pen_event.id == -1:
                new_pen_event.id = self.highest_id
                print('New PenEvent {}'.format(self.highest_id))
                self.highest_id += 1
            final_pen_events.append(new_pen_event)
        return final_pen_events

    def __calculate_distances_between_all_points(self, point_list_one, point_list_two):
        distances = []

        for i in range(len(point_list_one)):
            for j in range(len(point_list_two)):
                distance_between_points = distance.euclidean(point_list_one[i].get_coordinates(),
                                                             point_list_two[j].get_coordinates())
                distances.append([i, j, distance_between_points])

        # Sort list of lists by third element, in this case the distance between the points
        # https://stackoverflow.com/questions/4174941/how-to-sort-a-list-of-lists-by-a-specific-index-of-the-inner-list
        distances.sort(key=lambda x: x[2])

        return distances

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    # Simplified function when only one pen can be present at the time
    def merge_pen_events_single(self, new_pen_events):

        self.pen_events_to_remove = []

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
                    # self.stored_lines.append(np.array(active_pen_event.history))
                    self.stored_lines.append({active_pen_event.id: active_pen_event.history})

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
                    # self.stored_lines.append(np.array(last_pen_event.history))
                    self.stored_lines.append({last_pen_event.id: last_pen_event.history})
            else:

                new_pen_event.id = last_pen_event.id
                new_pen_event.first_appearance = last_pen_event.first_appearance
                new_pen_event.state_history = last_pen_event.state_history
                new_pen_event.state_history.append(new_pen_event.state)
                new_pen_event.last_seen_timestamp = now

                if HOVER_WINS:  # Overwrite the current state to hover
                    if new_pen_event.state != PenState.HOVER and PenState.HOVER in new_pen_event.state_history[
                                                                                   -NUM_CHECK_LAST_EVENT_STATES:]:
                        print(
                            'Pen Event {} has prediction {}, but State.HOVER is present in the last {} events, so hover wins'.format(
                                last_pen_event.id, new_pen_event.state, NUM_CHECK_LAST_EVENT_STATES))
                        new_pen_event.state = PenState.HOVER

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
        final_pen_events = self.__assign_new_ids(new_pen_events)

        for final_pen_event in final_pen_events:
            # Add current position to the history list, but ignore hover events
            if final_pen_event.state != PenState.HOVER:
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
                        PenState.HOVER) == NUM_HOVER_EVENTS_TO_END_LINE:
                    if DEBUG_MODE:
                        print('Pen Event {} turned from State.DRAG into State.HOVER'.format(final_pen_event.id))
                    if len(final_pen_event.history) > 0:
                        # self.stored_lines.append(np.array(final_pen_event.history))
                        self.stored_lines.append({final_pen_event.id: final_pen_event.history})

                    # print(final_pen_event.state, final_pen_event.state_history[-5:])
                    if DEBUG_MODE:
                        print('PEN Event {} ({} points) gets deleted because DRAW ended'.format(final_pen_event.id,
                                                                                                len(final_pen_event.history)))
                    final_pen_events = []

        self.active_pen_events = final_pen_events

        return self.active_pen_events, self.stored_lines, self.pen_events_to_remove

    # Assign a new ID to all PenEvents that have the placeholder ID "-1"


    # Function to merge the events, even when multiple pens are present at the same time
    def merge_pen_events_multiple_old(self, new_pen_events):
        now = round(time.time() * 1000)  # Get current timestamp

        # # Keep track of the current state
        # # TODO: Do this already when the pen event is created
        # for new_pen_event in new_pen_events:
        #     new_pen_event.state_history = [new_pen_event.state]

        # Iterate over copy of list
        # If a final_pen_event has been declared a "Click Event" in the last frame, this event is now over, and we can delete it.
        for active_pen_event in self.active_pen_events[:]:
            if active_pen_event.state == PenState.CLICK:
                self.__process_click_events(active_pen_event)

        # Compare all new_pen_events and active_pen_events and pair them by shortest distance to each other
        shortest_distance_point_pairs = self.__calculate_distances_between_all_points(self.active_pen_events,
                                                                                      new_pen_events, as_objects=True)

        for entry in shortest_distance_point_pairs:
            last_pen_event = self.active_pen_events[entry[0]]
            new_pen_event = new_pen_events[entry[1]]

            # We will reset the ID of all already paired events later. This check here will make sure that we do not
            # match an event multiple times
            if last_pen_event.id == -1:
                continue

            if new_pen_event.state == PenState.HOVER and len(last_pen_event.history) > 0 and PenState.DRAG not in last_pen_event.state_history[:-3]:
                # print('No State.DRAG for at least 3 frames')
                pass

            # TODO: Rework this check
            if new_pen_event.state == PenState.HOVER and PenState.HOVER not in last_pen_event.state_history[-3:]:
                # print('Pen Event {} turned from State.DRAG into State.HOVER'.format(last_pen_event.id))
                # new_pen_event.state_history.append(new_pen_event.state)
                # We now want to assign a new ID
                # TODO: Check why this event is called more than once
                # Maybe set state of old event to missing?
                continue
                # pass

            new_pen_event.state_history = last_pen_event.state_history
            new_pen_event.state_history.append(new_pen_event.state)

            # print(new_pen_event.state_history[-4:])

            # Move ID and other important information from the active touch final_pen_event into the new
            # touch final_pen_event
            if PenState.HOVER in new_pen_event.state_history[-4:-2] and not PenState.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:  #  last_pen_event.state == State.HOVER and new_pen_event.state != State.HOVER:
                pass
                # print('Pen Event {} turned from State.HOVER into State.DRAG'.format(last_pen_event.id))

            if HOVER_WINS:
                # Overwrite the current state to hover
                if new_pen_event.state != PenState.HOVER and PenState.HOVER in new_pen_event.state_history[-KERNEL_SIZE_HOVER_WINS:]:
                    # print('Pen Event {} has prediction {}, but State.HOVER is present in the last {} events, so hover wins'.format(last_pen_event.id, new_pen_event.state, KERNEL_SIZE_HOVER_WINS))
                    new_pen_event.state = PenState.HOVER
                else:
                    # print('Turning {} into a Drag event'.format(new_pen_event.state))
                    # if State.HOVER in new_pen_event.state_history[-4:-2]:
                    #     new_pen_event.state = State.NEW
                    # else:
                    # TODO: CHANGE this to allow for different types of drag events
                    # new_pen_event.state = State.DRAG  # last_pen_event.state
                    if new_pen_event.state != PenState.DRAG:
                        pass

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
                if active_pen_event.state == PenState.NEW:
                    # We detected a click event, but we do not remove it yet because it also could be a double click.
                    # We will check this the next time this function is called.
                    # print('Click event candidate found')
                    active_pen_event.state = PenState.CLICK
                    new_pen_events.append(active_pen_event)
                elif active_pen_event.state == PenState.DRAG:
                    # End of a drag event
                    # print('DRAG Event ended for Pen Event {}'.format(active_pen_event.id))
                    self.pen_events_to_remove.append(active_pen_event)
                    # print('Adding {} points of Event {} to the stored_lines list'.format(len(active_pen_event.history),
                    #                                                                      active_pen_event.id))
                    self.stored_lines.append(np.array(active_pen_event.history))
                    # self.new_pen_events.append(active_pen_event.history)
                elif active_pen_event.state == PenState.HOVER:
                    # End of a Hover event
                    # print('HOVER Event ended for Pen Event {}'.format(active_pen_event.id))
                    self.pen_events_to_remove.append(active_pen_event)

                    if len(active_pen_event.history) > 0:
                        # print('Adding {} points of Event {} to the stored_lines list'.format(
                        #     len(active_pen_event.history),
                        #     active_pen_event.id))
                        self.stored_lines.append(np.array(active_pen_event.history))


        # Now we have al list of new events that need their own unique ID. Those are assigned now
        final_pen_events = self.assign_new_ids(new_pen_events)

        for final_pen_event in final_pen_events:
            # Add current position to the history list, but ignore hover events
            if final_pen_event.state != PenState.HOVER:
                final_pen_event.history.append((final_pen_event.x, final_pen_event.y))

            # final_pen_event.state_history.append(final_pen_event.state)

            time_since_first_appearance = now - final_pen_event.first_appearance
            if final_pen_event.state != PenState.CLICK and final_pen_event.state != PenState.DOUBLE_CLICK and time_since_first_appearance > CLICK_THRESH_MS:
                if final_pen_event.state == PenState.NEW:
                    # Start of a drag event
                    print('DRAG Event started for Pen Event {}'.format(final_pen_event.id))
                    final_pen_event.state = PenState.DRAG
                # elif final_pen_event.state == State.HOVER:
                #     print('DETECTED Hover EVENT!')

        print(final_pen_events)

        return final_pen_events

    def __process_click_events_old(self, active_pen_event):
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



