import socket
import sys
import time
import base64
import random

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = '/tmp/sock.uds'
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
x_max = 1920
y_min = 0
y_max = 1080
line_id = 0
num_points = 0
x = int(x_max / 2)
y = int(y_max / 2)
delta = 20

color = (255, 255, 255)

while True:
    time.sleep(0.1)
    #time.sleep(0.01)
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

    sock.send(f'l {line_id} {color[0]} {color[1]} {color[2]} {x} {y} 1 |'.encode())
    print(f'{line_id} {x} {y}')

    #time.sleep(0.1)

    if num_points > 2:
        if random.randint(0, 20) == 0:
            sock.send(f'l {line_id} {color[0]} {color[1]} {color[2]} {x} {y} 0 |'.encode())
            line_id += 1
            num_points = 0
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
