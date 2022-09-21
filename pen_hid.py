# UInput is used to create a virtual input device
from evdev import UInput, ecodes as e, AbsInfo, InputDevice
import time

class InputSimulator:

    left_pressed = False
    right_pressed = False

    def __init__(self, w, h):
        # specify capabilities for our virtual input device
        self.capabilities = {
            e.EV_KEY: [e.KEY_POWER, e.BTN_LEFT, e.BTN_MOUSE, e.BTN_RIGHT, e.BTN_MIDDLE],
            e.EV_REL: [e.REL_WHEEL],
            e.EV_ABS: [
                    (e.ABS_X, AbsInfo(value=0, min=0, max=w, fuzz=0, flat=0, resolution=31)),
                    (e.ABS_Y, AbsInfo(0, 0, h, 0, 0, 31)),
                    (e.ABS_PRESSURE, AbsInfo(0, 0, 4000, 0, 0, 31))],
        }

        self.device = UInput(self.capabilities, name='mouse', version=0x3)
        self.was_pressed = False

    def sync_event(self):
        self.device.syn()
        #time.sleep(0.001)

    def input_event(self, x, y, state):
        self.device.write(e.EV_ABS, e.ABS_X, x)
        self.device.write(e.EV_ABS, e.ABS_Y, y)

        if state == 'draw':
            self.device.write(e.EV_KEY, e.BTN_LEFT, 1)
            self.was_pressed = True
        else:
            if self.was_pressed == True:
                self.device.write(e.EV_KEY, e.BTN_LEFT, 0)
                self.was_pressed = False

        self.device.syn()
        time.sleep(0.01)

    def move_event(self, x, y):
        self.device.write(e.EV_ABS, e.ABS_X, x)
        self.device.write(e.EV_ABS, e.ABS_Y, y)

    def click_event(self, btn, state):
        if btn == 'left':
            if self.right_pressed:
                print('right release')
                self.device.write(e.EV_KEY, e.BTN_RIGHT, 0)
                self.was_pressed = False
                self.right_pressed = False

            button = e.BTN_LEFT
        else:
            if self.left_pressed:
                print('left release')
                self.device.write(e.EV_KEY, e.BTN_LEFT, 0)
                self.was_pressed = False
                self.left_pressed = False

            button = e.BTN_RIGHT

        if state == 'draw':
            # print(btn + ' click')
            self.device.write(e.EV_KEY, button, 1)
            self.was_pressed = True
            if btn == 'left':
                self.left_pressed = True
            elif btn == 'right':
                self.right_pressed = True
        else:
            if self.was_pressed == True:
                print(btn + ' release')
                self.device.write(e.EV_KEY, button, 0)
                self.was_pressed = False

    def close(self):
        time.sleep(0.1)
        self.device.close()


if __name__ == '__main__':
    input_device = InputSimulator(1920, 1080)
    input_device.input_event(1, 0, 'draw')
    input_device.close()
