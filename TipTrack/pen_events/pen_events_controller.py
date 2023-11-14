
import time
import numpy as np
from scipy.spatial import distance

from TipTrack.pen_events.pen_state import PenState


NUM_HOVER_EVENTS_TO_END_LINE = 5  # We expect at least X Hover Events in a row to end that pen event

# Amount of time a pen can be missing until the pen event will be ended
TIME_POINT_MISSING_THRESHOLD_MS = 60  # 15

MAX_DISTANCE_FOR_MERGE = 500  # 200  # Maximum Distance between two points for them to be able to merge

# Simple Smoothing of the output points. 1 -> No smoothing; 0.5 -> Calculate a new point and use 50% of the previous
# point's and 50% of the new point's location.
SMOOTHING_FACTOR = 0.2  # Value between 0 and 1, depending on if the old or the new value should count more.

DEBUG_MODE = False


class PenEventsController:

    active_pen_events = []

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


