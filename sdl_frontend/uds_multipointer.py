import socket
import sys
import time
import base64
import random

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = '../uds_test'
print('connecting to %s' % server_address)
try:
    sock.connect(server_address)
except socket.error as msg:
    print(msg)
    sys.exit(1)

#message = "Hello World!"
#
#while(True):
#    try:
#        #sock.send(base64.encode(message))
#        sock.send(bytes(message, encoding='utf-8'))
#    except Exception as e:
#        print(e)
#    #response = sock.recv(64)
#    #print(response)
#    time.sleep(0.1)


counter = 0

x_min = 0
x_max = 3840
y_min = 0
y_max = 2160
line_id_1 = 0
line_id_2 = 1
num_points_1 = 0
num_points_2 = 0
x_1 = int(x_max / 2)
y_1 = int(y_max / 2)
x_2 = int(x_max / 2)
y_2 = int(y_max / 2)
delta = 20

state_1 = 0
state_2 = 0

color_1 = (255, 255, 255)
color_2 = (255, 255, 255)

while True:
    time.sleep(0.01)
    #time.sleep(0.01)
    num_points_1 += 1
    x_1 += random.randint(-delta, delta)
    y_1 += random.randint(-delta, delta)

    if x_1 < 0:
        x_1 = 0
    elif x_1 > x_max:
        x_1 = x_max

    if y_1 < 0:
        y_1 = 0
    elif y_1 > y_max:
        y_1 = y_max

    num_points_2 += 1
    x_2 += random.randint(-delta, delta)
    y_2 += random.randint(-delta, delta)

    if x_2 < 0:
        x_2 = 0
    elif x_2 > x_max:
        x_2 = x_max

    if y_2 < 0:
        y_2 = 0
    elif y_2 > y_max:
        y_2 = y_max

    sock.send(f'l {line_id_1} {color_1[0]} {color_1[1]} {color_1[2]} {x_1} {y_1} {state_1} |'.encode())
    sock.send(f'l {line_id_2} {color_2[0]} {color_2[1]} {color_2[2]} {x_2} {y_2} {state_2} |'.encode())

    print(f'{line_id_2} {x_2} {y_2}')
    print(f'{line_id_1} {x_1} {y_1}')

    #time.sleep(0.1)

    if num_points_1 > 2:
        if random.randint(0, 100) == 0:
            sock.send(f'l {line_id_1} {color_1[0]} {color_1[1]} {color_1[2]} {x_1} {y_1} 0 |'.encode())
            sock.send(f'f {line_id_1} |'.encode())
            line_id_1 += 1
            num_points_1 = 0
            state_1 = random.randint(0, 1)
            color_1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if num_points_2 > 2:
        if random.randint(0, 100) == 0:
            sock.send(f'l {line_id_2} {color_2[0]} {color_2[1]} {color_2[2]} {x_2} {y_2} 0 |'.encode())
            sock.send(f'f {line_id_2} |'.encode())
            line_id_2 += 1
            num_points_2 = 0
            state_2 = random.randint(0, 1)
            color_2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
