#!/usr/bin/python3

import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import threading

buf = []

fig = plt.figure()
ax = plt.gca()

def read_stdin():
    global buf
    for line in sys.stdin:
        print(line)
        line = line.strip('\n')
        try:
            i = float(line)
        except:
            continue
        buf.append(i)

        if(len(buf) > 200):
            buf.pop(0)

stdin_thread = threading.Thread(target=read_stdin)
stdin_thread.start()

def animate(i):
    global buf
    ax.clear()
    ax.plot(buf)
    
ani = animation.FuncAnimation(fig, animate, interval=1) 
plt.show()
