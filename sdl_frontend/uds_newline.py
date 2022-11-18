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
    time.sleep(0.1)
    #time.sleep(1)

    message = f'a 69 {int(0xFF0000)} 1,2;3,4;5,6 - |'
    print(message)
    sock.send(message.encode())
