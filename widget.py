from TipTrack.pen_events.pen_event import PenEvent

# A widget is a rectangular GUI element that can process user input.
class Widget():

    def __init__(self, id, pos_x, pos_y, width, height):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.height = height
        self.id = id

        self.is_visible = True
    
    def set_position(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def move(self, delta_x, delta_y):
        self.pos_x += delta_x
        self.pos_y = delta_y

    def set_dimensions(self, width, height):
        self.width = width
        self.height = height
    
    def is_point_on_widget(self, x, y):
        return x >= self.pos_x and x <= self.pos_x + self.width and y >= self.pos_y and y <= self.pos_y +self.height
    
    def set_visibility(self, is_visible):
        self.is_visible = is_visible

    
    # turns coordinates from the absolute coordinate system (origin in the upper left corner of the desk)
    # to coordinates in the widgets own coordinate system (origin in the widget's own upper left corner)
    def get_relative_coordinates(self,x, y):
        rel_x = x - self.pos_x
        rel_y = y - self.pos_y

        return rel_x, rel_y



    def on_click(self, pen_event):
        print(f"Widget with the id {self.id}: Subclass {__name__} has implemented no behavior for event on_click")

    
    def on_hover(self, pen_event):
        print(f"Widget with the id {self.id}: Subclass {__name__} has implemented no behavior for event on_hover")
    