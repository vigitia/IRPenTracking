import time
from TipTrack.pen_events.pen_state import PenState


class PenEvent:
    """ PenEvent

        Representation of a single pen event
    """

    def __init__(self, x, y, state=PenState.NEW):
        self.x = x  # Current x coordinate
        self.y = y
        self.id = -1  # ID of the PenEvent (-1 means that no ID has been assigned yet)

        # self.missing = False
        self.last_seen_timestamp = 0

        self.first_appearance = round(time.time() * 1000)
        self.state = state
        self.true_state = state
        self.history = []  # All logged x and y positions as tuples
        self.state_history = []  # All logged states (Hover, Draw, ...)

    def get_coordinates(self):
        return tuple([self.x, self.y])

    def __repr__(self):
        return 'PenEvent {} at ({}, {}) -> {}. Num Points: {}'.format(str(self.id), str(int(self.x)),
                                                                      str(int(self.y)), self.state, len(self.history))
