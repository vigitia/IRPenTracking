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
x_max = 1920
y_min = 0
y_max = 1080
line_id = 0
num_points = 0
x = int(x_max / 2)
y = int(y_max / 2)
delta = 20

while True:
    time.sleep(.05)
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

    sock.send(f'l {line_id} {x} {y} 1 '.encode())

    x1 = random.randint(x_min, x_max)
    y1 = random.randint(y_min, y_max)
    x2 = random.randint(x_min, x_max)
    y2 = random.randint(y_min, y_max)
    x3 = random.randint(x_min, x_max)
    y3 = random.randint(y_min, y_max)
    x4 = random.randint(x_min, x_max)
    y4 = random.randint(y_min, y_max)

    print(f'{line_id} {x} {y}')

    sock.send(f'r {line_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} 1 '.encode())
    
    if num_points > 2:
        if random.randint(0, 20) == 0:
            line_id += 1
            num_points = 0
