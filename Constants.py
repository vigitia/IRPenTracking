# Width and height of the output window/screen -> Target resolution
OUTPUT_WINDOW_WIDTH = 3840 #1920 #3840
OUTPUT_WINDOW_HEIGHT = 2160 #1080 #2160

# Width and height of the received frames
INPUT_FRAME_WIDTH = 1920  # 848
INPUT_FRAME_HEIGHT = 1200  # 480

# For Flir Blackfly S Calibration
CAM_EXPOSURE_FOR_CALIBRATION = 150000  # Increase Brightness to better see the corners
CAM_GAIN_FOR_CALIBRATION = 150  # Increase Brightness to better see the corners

# Eraser radius in pixels (assuming 4k)
ERASER_SIZE_SMALL = 10
ERASER_SIZE_BIG = 75

# Path to palette image
PALETTE_FILE_PATH = "assets/big_palette_expanded.png"

#Path to indicator image
INDICATOR_FILEPATH = "assets/palette_indicator.png"

#By default, white is at the 11th field of the palette (starting at 0)
POSITION_WHITE = 11

#dimensions of palette
UNSCALED_PALETTE_WIDTH = 1800
UNSCALED_PALETTE_HEIGHT = 150

#y-position of palette
UNSCALED_PALETTE_Y_POS = 0

# name of the unix socket
UNIX_SOCK_NAME = 'uds_test'
