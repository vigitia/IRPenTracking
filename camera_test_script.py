import PySpin
import cv2
import numpy as np

from pen_state import PenState
from ir_pen import IRPen

ir_pen = IRPen()

SERIAL_NUMBER_MASTER = str(22260470)
SERIAL_NUMBER_SLAVE = str(22260466)


system = PySpin.System.GetInstance()
blackFly_list = system.GetCameras()
print(len(system.GetCameras()))


camera0 = blackFly_list.GetBySerial(SERIAL_NUMBER_MASTER)
camera1 = blackFly_list.GetBySerial(SERIAL_NUMBER_SLAVE)

camera0.Init()
camera1.Init()
camera0.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
camera1.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
camera0.BeginAcquisition()
camera1.BeginAcquisition()

matrix0 = np.asarray([[-1.55739510e+00, -9.37397264e-02, 2.50047102e+03],
                     [2.52608449e-01, -1.77091818e+00, 1.53951289e+03],
                     [2.52775080e-04, 4.13147539e-05, 1.00000000e+00]])

matrix1 = np.asanyarray([[-1.35170720e+00, -3.53265982e-02, 2.36537791e+03],
                        [-5.86469873e-02, -1.17293975e+00, 1.30917670e+03],
                        [-1.67669502e-04, 5.80096166e-06, 1.00000000e+00]])

while True:
    image0 = camera0.GetNextImage()
    image1 = camera1.GetNextImage()

    image0 = image0.GetData().reshape(1200, 1920, 1)
    image1 = image1.GetData().reshape(1200, 1920, 1)

    # active_pen_events, stored_lines, _, _, rois = ir_pen.get_ir_pen_events([im_cv2_format], matrix)
    active_pen_events, stored_lines, _, _, rois = ir_pen.get_ir_pen_events_new([image0, image1], [matrix0, matrix1])

    cv2.imshow('Flir Blackfly S 0', image0)
    cv2.imshow('Flir Blackfly S 1', image1)
    # image0.release()

    cv2.waitKey(1)

# system.ReleaseInstance()