
# Source code for TipTrack

Precise, low-latency, robust optical pen tracking on arbitrary surfaces
using an IR-emitting pen tip.

![Figure 1: Drawing with the custom IR pen](assets/dragon.jpg)
Figure 1: Drawing with the custom IR pen


To build your own setup, you will need a computer running Linux, a projector and one or two infrared sensitive cameras.
If the cameras still let light through in the visible spectrum, please use IR cutoff filters (in our case 850nm).

Our current setup uses two _FLIR BFS-U3-23S3M-C_ cameras (1) with 850nm IR cutoff filters and an _Optoma UHZ50_ 4K projector (2)

![Hardware Setup](assets/setup.jpg)
Figure 2: Hardware setup

# Step 1: Install dependencies

## Install Python dependencies:

```
pip3 install opencv-python
pip3 install scikit-image
pip3 install tensorflow
```

For Flir BlackFly S cameras: Install Spinnaker SDK and PySpin by follwing the official guide: https://www.flir.com/products/spinnaker-sdk/

For Intel Realsense D435 cameras (not recommended because of low camera resolution):
```pip3 install pyrealsense2```


## Install dependencies for SDL frontend (optional)

If you want to use our sample low latency drawing application:

```
sudo apt install libsdl2-gfx-dev
sudo apt install libsdl2-image-dev
sudo apt install libsdl2-ttf-dev
```


# Step 2: Setup and calibrate Projector and Cameras

Mount the projector in a way so that it projects onto a flat surface (see Figure 2). The cameras should be placed in a way so that they can see the entire projection surface.

Set `CALIBRATION_MODE = True` in _flir_blackfly_s.py_ (or _realsense_d435.py_) and run the script.
A preview of each camera's feed will appear. These windows will help you to adjust the cameras. 

Afterwards, select the four corners of the projection area by clicking on them with your mouse cursor.

Finally, set `CALIBRATION_MODE = False`

# Step 3: Run the application

If you want to use the pen as an Input device: 
```python3 run_hid.py```

If you want to use our simple drawing application instead:
Start SDL frontend (low latency)

```
cd sdl_frontend
make
./sdl_frontend "../pipe_test" 0
```

Then start camera and pen tracking by running
```python3 ir_pen.py```



