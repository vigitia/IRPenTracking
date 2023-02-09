import os

import PySpin
import cv2
import numpy as np

from pen_state import PenState
from ir_pen import IRPen
from surface_extractor import SurfaceExtractor

ir_pen = IRPen()

SERIAL_NUMBER_MASTER = str(22260470)
SERIAL_NUMBER_SLAVE = str(22260466)

FRAME_WIDTH = 1920  # 800  # 1920
FRAME_HEIGHT = 1200  # 600  # 1200
EXPOSURE_TIME_MICROSECONDS = 600  # μs -> must be lower than the frame time (FRAMERATE / 1000)
GAIN = 18  # Controls the amplification of the video signal in dB.
FRAMERATE = 158  # Target number of Frames per Second (Min: 1, Max: 158)


system = PySpin.System.GetInstance()
blackFly_list = system.GetCameras()
print(len(system.GetCameras()))


camera0 = blackFly_list.GetBySerial(SERIAL_NUMBER_MASTER)
camera1 = blackFly_list.GetBySerial(SERIAL_NUMBER_SLAVE)

camera0.Init()
camera1.Init()

camera0.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
camera1.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

camera0.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
camera1.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

camera0.ExposureTime.SetValue(max(camera0.ExposureTime.GetMin(), min(camera0.ExposureTime.GetMax(), float(EXPOSURE_TIME_MICROSECONDS))))
camera1.ExposureTime.SetValue(max(camera1.ExposureTime.GetMin(), min(camera1.ExposureTime.GetMax(), float(EXPOSURE_TIME_MICROSECONDS))))

camera0.GainAuto.SetValue(PySpin.GainAuto_Off)
camera1.GainAuto.SetValue(PySpin.GainAuto_Off)

camera0.Gain.SetValue(max(camera0.Gain.GetMin(), min(camera0.Gain.GetMax(), float(GAIN))))
camera1.Gain.SetValue(max(camera1.Gain.GetMin(), min(camera1.Gain.GetMax(), float(GAIN))))

camera0.AcquisitionFrameRateEnable.SetValue(True)
camera1.AcquisitionFrameRateEnable.SetValue(True)

camera0.AcquisitionFrameRate.SetValue(min(camera0.AcquisitionFrameRate.GetMax(), FRAMERATE))
camera1.AcquisitionFrameRate.SetValue(min(camera1.AcquisitionFrameRate.GetMax(), FRAMERATE))

camera0.BeginAcquisition()
camera1.BeginAcquisition()

# matrix0 = np.asanyarray([[-1.55739510e+00, -9.37397264e-02, 2.50047102e+03],
#                      [2.52608449e-01, -1.77091818e+00, 1.53951289e+03],
#                      [2.52775080e-04, 4.13147539e-05, 1.00000000e+00]])
#
# matrix1 = np.asanyarray([[-1.35170720e+00, -3.53265982e-02, 2.36537791e+03],
#                         [-5.86469873e-02, -1.17293975e+00, 1.30917670e+03],
#                         [-1.67669502e-04, 5.80096166e-06, 1.00000000e+00]])

surface_extractor = SurfaceExtractor()

matrix0 = surface_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(SERIAL_NUMBER_MASTER))
matrix1 = surface_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(SERIAL_NUMBER_SLAVE))

image_id = 0

while True:
    image0 = camera0.GetNextImage()
    image1 = camera1.GetNextImage()

    image0 = image0.GetData().reshape(1200, 1920, 1)
    image1 = image1.GetData().reshape(1200, 1920, 1)

    # image0 = image0.GetNDArray()
    # image1 = image1.GetNDArray()

    active_pen_events, stored_lines, pen_events_to_remove = ir_pen.get_ir_pen_events_new([image0, image1], [matrix0, matrix1])

    cv2.imshow('Flir Blackfly S 0', image0)
    cv2.imshow('Flir Blackfly S 1', image1)

    # image0.release()
    # image1.release()

    key = cv2.waitKey(1)
    if key == 32:  # Spacebar
        cv2.imwrite(os.path.join('screenshots', 'Flir_Blackfly_S_{}_{}.png'.format(SERIAL_NUMBER_MASTER, image_id)), image0)
        cv2.imwrite(os.path.join('screenshots', 'Flir_Blackfly_S_{}_{}.png'.format(SERIAL_NUMBER_SLAVE, image_id)), image1)
        image_id += 1

# system.ReleaseInstance()
