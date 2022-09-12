import os, time, sys
import random

pipe_name = 'test_fifo'

if not os.path.exists(pipe_name):
    os.mkfifo(pipe_name)  

pipeout = os.open(pipe_name, os.O_WRONLY)

counter = 0

x_min = 0
x_max = 1920
y_min = 0
y_max = 1080
line_id = 0
num_points = 0
x = int(x_max / 2)
y = int(y_max / 2)
delta = 20

while True:
    time.sleep(0.01)
    num_points += 1
    x += random.randint(-delta, delta)
    y += random.randint(-delta, delta)

    if x < 0:
        x = 0
    elif x > x_max:
        x = x_max

    if y < 0:
        y = 0
    elif y > y_max:
        y = y_max

    #x = random.randint(x_min, x_max)
    #y = random.randint(y_min, y_max)

    os.write(pipeout, f'{line_id} {x} {y} 1 '.encode())
    print(f'{line_id} {x} {y}')

    if num_points > 2:
        if random.randint(0, 20) == 0:
            line_id += 1
            num_points = 0
