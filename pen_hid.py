# UInput is used to create a virtual input device
from evdev import UInput, ecodes as e, AbsInfo, InputDevice
import time

# specify capabilities for our virtual input device
capabilities = {
    e.EV_KEY: [e.KEY_POWER, e.BTN_LEFT, e.BTN_MOUSE, e.BTN_RIGHT, e.BTN_MIDDLE],
    e.EV_REL: [e.REL_WHEEL],
    e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=0, min=0, max=1920, fuzz=0, flat=0, resolution=31)),
            (e.ABS_Y, AbsInfo(0, 0, 1080, 0, 0, 31)),
            (e.ABS_PRESSURE, AbsInfo(0, 0, 4000, 0, 0, 31))],
}

device = UInput(capabilities, name='mouse', version=0x3)
was_pressed = False

def input_event(x, y, state):
    global was_pressed, device
    device.write(e.EV_ABS, e.ABS_X, x)
    device.write(e.EV_ABS, e.ABS_Y, y)

    if state == 'draw':
        device.write(e.EV_KEY, e.BTN_LEFT, 1)
        was_pressed = True
    else:
        if was_pressed == True:
            device.write(e.EV_KEY, e.BTN_LEFT, 0)
            was_pressed = False

    device.syn()
    time.sleep(0.01)

def close():
    global device
    time.sleep(0.1)
    device.close()

if __name__ == '__main__':
    input_event(1, 0, 'draw')
    close()
