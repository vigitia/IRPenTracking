import sys
import time
import threading
from pyaxidraw import axidraw
import signal
import pandas as pd

from TipTrack.pen_events.ir_pen import IRPen
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS

class CamController(threading.Thread):

    coords = (0,0)

    alive = True

    def __init__(self):
        super().__init__()
        self.ir_pen = IRPen()

    def run(self):
        flir_blackfly_s = FlirBlackflyS(subscriber=self)
        while self.alive:
            pass

        print('FINISHED')
        sys.exit(0)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(frames) > 0:
            # print('Frames!')
            active_pen_events, stored_lines, _, _, debug_distances, rois = self.ir_pen.get_ir_pen_events_multicam(
                frames, matrices)

            if len(active_pen_events) > 0:
                self.coords = (active_pen_events[0].x, active_pen_events[0].y)


class Run:

    min_x = 0
    min_y = 0
    max_x = 100
    max_y = 100

    cur_x = 0
    cur_y = 0
    cur_angle = 0

    # total_height = 190
    # total_width = 270
    total_height = 120
    total_width = 210
    direction = 1

    delay = 1.0  # 0.2  # 0.15

    max_dist = 256
    num_trials = 12
    # 256 128 64 32 16 8 4 2 1 0.5 0.25 # 0.125 0.0625 0.03125 0.015625

    distances = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]

    data_list = []

    def __init__(self):


        self.ad = axidraw.AxiDraw()

        self.ad.interactive()
        self.ad.connect()
        self.ad.options.units = 2 # millimeters

        self.ad.options.const_speed = True

        self.ad.options.pen_pos_down = 0
        self.ad.options.pen_pos_up = 40
        self.ad.update()

        signal.signal(signal.SIGINT, self.handle_interrupt_signal)

        current_dist = self.max_dist

        self.ad.penup()
        self.move(0, 100)
        self.ad.pendown()

        # thread = threading.Thread(target=CamController)
        # thread.start()

        self.cam_controller = CamController()
        self.cam_controller.start()

        time.sleep(10)

        self.loop()

        # while True:
        #     self.ad.moveto(0, 0)
        #     time.sleep(1)
        #     self.ad.lineto(0, 0)
        #     time.sleep(1)

    def loop(self):
        for current_dist in self.distances:
            for i in range(25):
                # ad.pendown()
                time.sleep(self.delay)
                coords_ir_pen = self.cam_controller.coords
                print(coords_ir_pen)
                coords_axidraw = (i * current_dist, 100)
                self.data_list.append(
                    {'dist': current_dist, 'num': i, 'axi_x': coords_axidraw[0], 'axi_y': coords_axidraw[1],
                     'ir_x': coords_ir_pen[0], 'ir_y': coords_ir_pen[1]})
                self.move(i * current_dist, 100)
            time.sleep(self.delay)
            self.ad.penup()
            self.move(0, 100)
            self.ad.pendown()

        df = pd.DataFrame(self.data_list)
        df.to_csv('resolution_data.csv')

        self.ad.penup()
        self.move(0, 0)
        time.sleep(1)
        self.cam_controller.alive = False
        sys.exit(0)

    def move(self, x, y):
        # global cur_x, cur_y
        self.cur_x = x
        self.cur_y = y
        self.ad.goto(x, y)

    def draw(self, x, y):
        # global cur_x, cur_y
        self.cur_x = x
        self.cur_y = y
        self.ad.lineto(x, y)

    # close the program softly when ctrl+c is pressed
    def handle_interrupt_signal(self, signal, frame):
        self.move(0, 0)
        time.sleep(1)
        sys.exit(0)

    # def test_run(self):
    #     global cur_x, cur_y
    #     self.move(0, self.total_height)
    #     time.sleep(1)
    #
    #     self.move(self.total_width, self.total_height)
    #     time.sleep(1)
    #
    #     self.move(self.total_width, 0)
    #     time.sleep(1)
    #
    #     self.move(0, 0)
    #     time.sleep(1)

    #test_run()
    #time.sleep(1)
    #sys.exit(0)

    ## here



##


#for x in range(0, total_width, step_size):
#    for y in range(0, total_height, step_size):
#        cur_y += step_size * direction
#        move(cur_x, cur_y)
#        time.sleep(delay)
#        for a in range(0, 90, 5):
#            rotate(a)
#            record_img()
#    cur_x += step_size
#
#    move(cur_x, cur_y)
#    time.sleep(1)
#    for a in range(0, 90, 5):
#        rotate(a)
#        record_img()
#
#    direction *= -1


if __name__ == '__main__':
    run = Run()

