import socket
import sys
import time
import base64
import random
import os

from pyaxidraw import axidraw
import signal

ad = axidraw.AxiDraw()

ad.interactive()
ad.connect()
ad.options.units = 2 # millimeters

ad.options.const_speed = True

ad.options.pen_pos_down = 0
ad.options.pen_pos_up = 40
ad.options.speed_pendown = 100
ad.update()

ad.penup()

last_state = 0

total_height = 120
total_width = 210

width_factor = total_width / 3840
height_factor = total_height / 2160

def handle_interrupt_signal(signal, frame):
    ad.penup()
    ad.goto(0, 0)
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt_signal)

ad.goto(0, 0)
time.sleep(1)

print('go')

server_address = 'uds_test'

try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

sock.bind(server_address)

sock.listen(1)

connection, client_address = sock.accept()

while True:
    data = connection.recv(1024)
    print(data)
    data_chunks = data.decode().split('l ')
    print(len(data_chunks))

    for i in range(len(data_chunks)):
        data_split = data_chunks[-i].split(' ')
        if len(data_split) >= 5:
            print(data_split)

            id, x, y, state, _ = data_split[:5]

            x = int(x)
            y = int(y)
            state = int(state)
            print(x, y, state)


            if state != last_state:
                if state == 1:
                    ad.pendown()
                else:
                    ad.penup()
                last_state = state

            ad.goto(x * width_factor, y * height_factor)
            #time.sleep(0.025)
            break