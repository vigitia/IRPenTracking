# Using an Infrared Pen as an Input Device for Projected Augmented Reality Tabletops 

Source code for our paper "Using an Infrared Pen as an Input Device for Projected Augmented Reality Tabletops "

# Install Python dependencies:

TODO

```
pip3 install opencv-python
pip3 install scikit-image
pip3 install tensorflow

```

For RealsenseD435:
pip3 install pyrealsense2

For Flir BlackFly S
TODO: Add guide for install Spinnaker SDK

For HID Layer:
sudo aot

# Install SDL

```
sudo apt install libsdl2-gfx-dev
sudo apt install libsdl2-image-dev
sudo apt install libsdl2-ttf-dev
```



# Calibrate Projector and Camera

Set the flag "CALIBRATION_MODE" in _realsense_d435.py_ to True.
Run _realsense_d435.py_. A window will appear. Select the four corners of the projection.
Set the Flag back to false.

# Start SDL frontend (low latency)

```
cd sdl_frontend
make
./sdl_frontend "../pipe_test" 0
```

# Start camera and pen tracking

```python3 ir_pen.py```

