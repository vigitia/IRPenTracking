import sys
import time
import threading
from matplotlib import pyplot as plt
from pyaxidraw import axidraw
import signal
import pandas as pd

ad = axidraw.AxiDraw()

ad.interactive()
ad.connect()
ad.options.units = 2 # millimeters

ad.options.const_speed = True

ad.options.pen_pos_down = 0
ad.options.pen_pos_up = 40
ad.update()

min_x = 0
min_y = 0
max_x = 100
max_y = 100

cur_x = 0
cur_y = 0
cur_angle = 0

#total_height = 190
#total_width = 270
total_height = 120
total_width = 210
direction = 1

delay = 0.2 # 0.15

max_dist = 256
num_trials = 12
# 256 128 64 32 16 8 4 2 1 0.5 0.25 # 0.125 0.0625 0.03125 0.015625

distances = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]

data_list = []

def move(x, y):
    global cur_x, cur_y
    cur_x = x
    cur_y = y
    ad.goto(x, y)

def draw(x, y):
    global cur_x, cur_y
    cur_x = x
    cur_y = y
    ad.lineto(x, y)

# close the program softly when ctrl+c is pressed
def handle_interrupt_signal(signal, frame):
    move(0, 0)
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt_signal)

def test_run():
    global cur_x, cur_y
    move(0, total_height)
    time.sleep(1)

    move(total_width, total_height)
    time.sleep(1)

    move(total_width, 0)
    time.sleep(1)

    move(0, 0)
    time.sleep(1)

#test_run()
#time.sleep(1)
#sys.exit(0)

## here

def get_ir_pen_coords():
    pass

current_dist = max_dist

ad.penup()
move(0, 100)
ad.pendown()

for current_dist in distances:
    for i in range(25):
        #ad.pendown()
        time.sleep(delay)
        coords_ir_pen = get_ir_pen_coords()
        coords_axidraw = (i * current_dist, 100)
        data_list.append({'dist' : current_dist, 'num' : i, 'axi_x' : coords_axidraw[0], 'axi_y' : coords_axidraw[1], 'ir_x' : coords_ir_pen[0], 'ir_y' : coords_ir_pen[1]})
        move(i * current_dist, 100)
    time.sleep(delay)
    ad.penup()
    move(0, 100)
    ad.pendown()

df = pd.DataFrame(data_list)
df.to_csv('resolution_data.csv')

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

ad.penup()
move(0, 0)
time.sleep(1)
sys.exit(0)

while True:
    ad.moveto(0, 0)
    time.sleep(1)
    ad.lineto(0, 0)
    time.sleep(1)
