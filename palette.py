from widget import Widget

class Palette (Widget):

    #@params: fields: an array of tuples of length 3. Tuples contain RGB color values. Negative values are reserved for special codes: For example, a value of (-1, -1, -1) doesn't select a color, but the ERASE-tool instead.
    #   ( r,  g,  b) = red, green and blue values
    #   (-1, -1, -1) = switch to erase tool
    #   (-2, -2, -2), (-3, -3, -3) etc... reserved for other actions (when we need them implemented)s
    def __init__(self,id,x, y,fields, field_size, callback):
        super().__init__(id, x, y, len(fields) * field_size, field_size)

        self.fields = fields
        self.field_size = field_size
        self.callback = callback
    
    # Selects a color (or a tool).
    # returns: A 2-tuple with an action (currently "COLOR" or "ERASE") and a color code (if applicable)
    def on_click(self, pen_event):
        rel_x, rel_y = super().get_relative_coordinates(*pen_event.get_coordinates())
        print(f"Click event at {rel_x}, {rel_y}")
        idx_x = int(rel_x / self.field_size)

        color_code = self.fields[idx_x]

        if color_code[0] >= 0 and color_code[1] >= 0 and color_code[2] >= 0: #no negative values => it's actually a color
            self.callback("COLOR", color_code)
        elif color_code[0] == -1:
            self.callback("ERASE", None)
            




        



    
