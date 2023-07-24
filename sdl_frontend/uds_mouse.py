import socket
import sys
import time
import base64
import random
from pynput import mouse

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
if len(sys.argv) > 1:
    server_address = sys.argv[1]
else:
    server_address = '../uds_test'

print('connecting to %s' % server_address)
try:
    sock.connect(server_address)
except socket.error as msg:
    print(msg)
    sys.exit(1)

counter = 0

x_min = 0
x_max = 1920 #3840
y_min = 0
y_max = 1080 #2160

color_1 = (255, 0, 0)

def send_message(msg):
    try:
        msg_encoded = bytearray(msg, 'ascii')
        size = len(msg_encoded)
        sock.send(size.to_bytes(4, 'big'))
        sock.send(msg_encoded) # , MSG_NOSIGNAL
    except socket.error:
        print('socket error')
        sys.exit(0)
    except BrokenPipeError:
        print('broken pipe')
        sys.exit(0)
    except Exception as e:
        print('other error', e)
        sys.exit(0)

line_counter = 0

mouse_pressed = False

line_id = 0

def on_move(x, y):
    global mouse_pressed, color_1, line_id
    x = x % x_max
    state = 1 if mouse_pressed else 0
    send_message(f'l {line_id} {color_1[0]} {color_1[1]} {color_1[2]} {x} {y} {state}')

def on_click(x, y, button, pressed):
    global mouse_pressed, color_1, line_id
    if pressed is not mouse_pressed and pressed == False:
        # release
        send_message(f'f {line_id}')
        line_id += 1
        #color_1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    mouse_pressed = pressed

listener = mouse.Listener(on_move=on_move, on_click=on_click)
listener.start()

while True:
    time.sleep(0.0001)
