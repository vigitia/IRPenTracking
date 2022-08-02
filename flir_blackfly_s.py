#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import sys
import threading
import time

import PySpin
import cv2
import numpy as np

from surface_selector import SurfaceSelector
from table_extraction_service import TableExtractionService

# CONSTANTS and Camera Settings

DEBUG_MODE = True
CALIBRATION_MODE = False
EXTRACT_PROJECTION_AREA = False
SHOW_DEBUG_STACKED_FRAMES = False

CALIBRATION_DATA_PATH = ''

FRAME_WIDTH = 1920  # 800  # 1920
FRAME_HEIGHT = 1200  # 600  # 1200
EXPOSURE_TIME_MICROSECONDS = 400  # μs -> must be lower than the frame time (FRAMERATE / 1000)
GAIN = 18  # Controls the amplification of the video signal in dB.
FRAMERATE = 158  # Target number of Frames per Second (Min: 1, Max: 158)
NUM_BUFFERS = 1  # Number of image buffers per camera

# Set the Serial Numbers of the primary and secondary cameras. Needed for the hardware trigger
SERIAL_NUMBER_MASTER = str(22260470)
SERIAL_NUMBER_SLAVE = str(22260466)


cam_image_master = None
cam_image_slave = None

all_cameras_ready = False

matrices = [[]]


table_extractor = TableExtractionService()


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """
    This class defines the properties, parameters, and the event handler itself.
    """

    start_time = time.time()
    frame_counter = 0

    def __init__(self, event_name, cam_id, cam, subscriber):
        super(DeviceEventHandler, self).__init__()
        self.event_name = event_name
        self.count = 0
        self.cam_id = cam_id
        self.cam = cam
        self.subscriber = subscriber

    def OnDeviceEvent(self, event_name):
        """
        Callback function when a device event occurs.
        Note event_name is a wrapped gcstring, not a Python string, but basic operations such as printing and comparing
        with Python strings are supported.
        """

        global all_cameras_ready
        if not all_cameras_ready:
            return

        # Check if we receive the expected event type
        if event_name == self.event_name:
            self.count += 1

            # Print information on specified device event
            # print('\tDevice Event "{}" ({}) from camera {}; {}'.format(event_name, self.GetDeviceEventId(), self.cam_id,
            #                                                            self.count))

            # start_time = datetime.datetime.now()
            image_result = self.cam.GetNextImage(1000)

            # print(image_result.GetTimeStamp() / 1e9)

            if image_result.IsIncomplete():  # Ensure image completion
                print('Image incomplete with image status %d' % image_result.GetImageStatus())
            else:
                if self.cam_id == SERIAL_NUMBER_MASTER:
                    global cam_image_master
                    cam_image_master = image_result.GetNDArray()
                    # cam_image_master = cv2.remap(cam_image_master, self.rectify_maps[i][0], self.rectify_maps[i][1], interpolation=cv2.INTER_LINEAR)
                elif self.cam_id == SERIAL_NUMBER_SLAVE:
                    global cam_image_slave
                    cam_image_slave = image_result.GetNDArray()
                    # cam_image_master = cv2.remap(cam_image_master, self.rectify_maps[i][0], self.rectify_maps[i][1], interpolation=cv2.INTER_LINEAR)

            #  Images retrieved directly from the camera need to be released in order to keep from filling the buffer.
            image_result.Release()

            # end_time = datetime.datetime.now()
            # run_time = (end_time - start_time).microseconds / 1000.0
            # print('Time for self.cam.GetNextImage(1000)', run_time, self.cam_id)

            self.check_both_frames_available()

        else:
            # Print no information on non-specified event
            print('\tDevice event occurred; not %s; ignoring...' % self.event_name)

    def check_both_frames_available(self):
        global cam_image_master
        global cam_image_slave

        if cam_image_master is not None and cam_image_slave is not None:
            # self.frame_counter += 1

            # print('Got both frames')
            if self.subscriber is not None:
                global matrices
                camera_serial_numbers = [SERIAL_NUMBER_MASTER, SERIAL_NUMBER_SLAVE]
                self.subscriber.on_new_frame_group([cam_image_master, cam_image_slave], camera_serial_numbers, matrices)

            # Reset variables
            cam_image_master = None
            cam_image_slave = None

            # if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
            #     if DEBUG_MODE:
            #         print("FPS: %s (Warning: DEBUG_MODE might reduce FPS)" % round(self.frame_counter / (time.time() - self.start_time), 1))
            #     else:
            #         print("FPS: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
            #     self.frame_counter = 0
            #     self.start_time = time.time()


class FlirBlackflyS:

    camera_matrices = []
    dist_matrices = []
    rectify_maps = []

    system = None
    cam_list = []

    device_event_handlers = []

    device_serial_numbers = []

    # newest_frames = []
    # matrices = []
    # new_frames_available = False

    start_time = time.time()
    frame_counter = 0

    def __init__(self, cam_exposure=EXPOSURE_TIME_MICROSECONDS, framerate=FRAMERATE, subscriber=None):
        global EXPOSURE_TIME_MICROSECONDS
        global FRAMERATE
        EXPOSURE_TIME_MICROSECONDS = cam_exposure
        FRAMERATE = framerate

        # if not CALIBRATION_MODE:
        #     self.table_extractor = TableExtractionService()

        self.init_cameras(subscriber)

    def init_cameras(self, subscriber):
        self.system = PySpin.System.GetInstance()  # Retrieve singleton reference to system object

        if DEBUG_MODE:
            version = self.system.GetLibraryVersion()  # Get current library version
            print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        self.cam_list = self.system.GetCameras()  # Retrieve list of cameras from the system

        num_cameras = self.cam_list.GetSize()
        print('Number of cameras detected: %d' % num_cameras)

        self.load_camera_calibration_data(num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            self.cam_list.Clear()  # Clear camera list before releasing system
            self.system.ReleaseInstance()  # Release system instance
            print('No cameras found')
            return False

        try:
            if DEBUG_MODE:
                # Retrieve transport layer nodemaps and print device information for each camera
                for i, cam in enumerate(self.cam_list):
                    nodemap_tldevice = cam.GetTLDeviceNodeMap()  # Retrieve Transport layer device nodemap
                    self.print_device_info(nodemap_tldevice, i)  # Print device information

            # Initialize each camera
            for i, cam in enumerate(self.cam_list):
                # Retrieve device serial numbers
                device_serial_number = -1
                node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
                if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                    device_serial_number = node_device_serial_number.GetValue()
                    print('Camera %d serial number: %s' % (i, device_serial_number))
                self.device_serial_numbers.append(device_serial_number)

                if not CALIBRATION_MODE:
                    global matrices
                    global table_extractor

                    if device_serial_number == SERIAL_NUMBER_MASTER:
                        matrices[0] = table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(device_serial_number))
                    else:
                        matrices.append(table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(device_serial_number)))

                    # matrices[device_serial_number] = table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(device_serial_number))
                    # matrices.append(table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Blackfly S {}'.format(device_serial_number)))

                cam.Init()  # Initialize camera

                self.init_hardware_trigger(cam, device_serial_number)

                success, device_event_handler = self.configure_device_events(cam, device_serial_number, subscriber)
                if not success:
                    print('Error in configure_device_events()')
                    time.sleep(10)
                self.device_event_handlers.append(device_event_handler)

                success = self.apply_camera_settings(cam, device_serial_number)
                if not success:
                    print('Errors while applying settings to the cameras')
                    time.sleep(10)

            cam_slave = self.cam_list.GetBySerial(SERIAL_NUMBER_SLAVE)
            cam_master = self.cam_list.GetBySerial(SERIAL_NUMBER_MASTER)

            cam_slave.BeginAcquisition()  # Begin acquiring images
            cam_master.BeginAcquisition()  # Begin acquiring images
            global all_cameras_ready
            all_cameras_ready = True

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

    def init_hardware_trigger(self, cam, device_serial_number):
        if device_serial_number == SERIAL_NUMBER_MASTER:  # Set Master
            print('Set Master')
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            cam.LineMode.SetValue(PySpin.LineMode_Output)
            cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
            cam.V3_3Enable.SetValue(True)
        elif device_serial_number == SERIAL_NUMBER_SLAVE:
            print('Set Slave')
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            # Set TriggerActivation Falling Edge TODO: better Rising Edge? ...
            cam.TriggerActivation.SetValue(PySpin.TriggerActivation_FallingEdge)
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        else:
            print('WRONG CAMERA ID FOR MASTER AND SLAVE!:', type(device_serial_number))

        # if device_serial_number == SERIAL_NUMBER_MASTER:  # Set Master
        #     print('Set Master')
        #
        #     cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        #     # cam.LineMode.SetValue(PySpin.LineMode_Output)
        #     cam.V3_3Enable.SetValue(True)
        #     # TODO: TriggerType Software für Master
        # elif device_serial_number == SERIAL_NUMBER_SLAVE:
        #     # Set up secondary camera trigger
        #     print('Set Slave')
        #     cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        #     cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        #     cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        #     cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        #     # TODO: TriggerType Hardware für Slave
        # else:
        #     print('WRONG CAMERA ID FOR MASTER AND SLAVE!:', type(device_serial_number))

    def load_camera_calibration_data(self, num_cameras):
        print('load calibration data')

        for i in range(num_cameras):
            self.camera_matrices.append([None])
            self.dist_matrices.append([None])
            try:
                # TODO: Change path
                cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, 'FlirBlackflyS {}.yml'.format(i)),
                                          cv2.FILE_STORAGE_READ)
                self.camera_matrices[i] = cv_file.getNode('K').mat()
                self.dist_matrices[i] = cv_file.getNode('D').mat()

                map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrices[i], self.dist_matrices[i], None, None,
                                                         (FRAME_WIDTH, FRAME_HEIGHT), cv2.CV_32FC1)

                self.rectify_maps.append([])
                self.rectify_maps[i] = [map1, map2]

                cv_file.release()
            except Exception as e:
                print(e)
                print('Cant load calibration data for FlirBlackflyS {}.yml'.format(i))

    def apply_camera_settings(self, cam, serial_number):

        # TODO: Also set GAIN, WIDTH AND HEIGHT, X_OFFSET, Y_OFFSET

        # Set Acquisition Mode to Continuous: acquires images continuously
        if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            print("PySpin:Camera:AcquistionMode: {}".format(cam.AcquisitionMode.GetValue()))
        else:
            print("PySpin:Camera:AcquisionMode: no access")
            return False

        # Set Autoexposure off
        if cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            print("PySpin:Camera:ExposureAuto: {}".format(cam.ExposureAuto.GetValue()))
        else:
            print("PySpin:Camera:Failed to set Autoexposure to off")
            return False

        # Check if Autoexposure is off and exposure mode is set to "Timed" (needed for manually setting exposure)
        if cam.ExposureMode.GetValue() != PySpin.ExposureMode_Timed:
            print("PySpin:Camera:Can not set exposure! Exposure Mode needs to be Timed")
            return False
        if cam.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
            print("PySpin:Camera:Can not set exposure! Exposure is Auto")
            return False

        # Set Exposure Time (in microseconds).
        # Exposure time should not be greater than frame time, this would reduce the resulting framerare
        if cam.ExposureTime.GetAccessMode() == PySpin.RW:
            cam.ExposureTime.SetValue(
                max(cam.ExposureTime.GetMin(), min(cam.ExposureTime.GetMax(), float(EXPOSURE_TIME_MICROSECONDS))))
            print("PySpin:Camera:Exposure:{}".format(cam.ExposureTime.GetValue()))
        else:
            print("PySpin:Camera:Failed to set exposure to:{}".format(EXPOSURE_TIME_MICROSECONDS))
            return False

        # Set GainAuto to off.
        if cam.GainAuto.GetAccessMode() == PySpin.RW:
            cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            print("PySpin:Camera:GainAuto: {}".format(cam.GainAuto.GetValue()))
        else:
            print("PySpin:Camera:Failed to set GainAuto to off")
            return False

        # Set Gain. Controls the amplification of the video signal in dB.
        if cam.Gain.GetAccessMode() == PySpin.RW:
            cam.Gain.SetValue(
                max(cam.Gain.GetMin(), min(cam.Gain.GetMax(), float(GAIN))))
            print("PySpin:Camera:Gain:{}".format(cam.Gain.GetValue()))
        else:
            print("PySpin:Camera:Failed to set Gain to:{}".format(GAIN))
            return False

        # TODO: Check if manually setting the Black Level has any impact on our output frames

        # Set Acquisiton Frame Rate only for Master camera
        if serial_number == SERIAL_NUMBER_MASTER:
            # Set Acquisiton Frame Rate Enable = True to be able to manually set the framerate
            if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRateEnable.SetValue(True)
                print("PySpin:Camera:AcquisionFrameRateEnable: {}".format(cam.AcquisitionFrameRateEnable.GetValue()))
            else:
                print("PySpin:Camera:AcquisionFrameRateEnable: no access")
                return False

            # Set Camera Acquisition Framerate
            if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRate.SetValue(min(cam.AcquisitionFrameRate.GetMax(), FRAMERATE))
                print('PySpin:Camera:CameraAcquisitionFramerate:', cam.AcquisitionFrameRate.GetValue())
            else:
                print("PySpin:Camera:Failed to set CameraAcquisitionFramerate to:{}".format(FRAMERATE))
                return False

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(
                stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        # Display Buffer Info
        print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        print('Default Buffer Count: %d' % buffer_count.GetValue())
        print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        buffer_count.SetValue(NUM_BUFFERS)

        print('Buffer count set to: %d' % buffer_count.GetValue())

        # handling_mode_entry = handling_mode.GetEntryByName('NewestFirst')
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')

        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        print('Buffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())

        # if cam.Width.GetAccessMode() == PySpin.RW:
        #     cam.Width.SetValue(1920)
        #     print("PySpin:Camera:Width:{}".format(cam.Width.GetValue()))
        # else:
        #     print("PySpin:Camera:Failed to set Width to:{}".format(1920))
        #     return False
        #
        # if cam.Height.GetAccessMode() == PySpin.RW:
        #     cam.Height.SetValue(1200)
        #     print("PySpin:Camera:Height:{}".format(cam.Height.GetValue()))
        # else:
        #     print("PySpin:Camera:Failed to set Height to:{}".format(1200))
        #     return False

        # node_width = PySpin.CIntegerPtr(nodemap_tldevice.GetNode('Width'))
        # node_width.SetValue(800)
        # if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
        #     width_to_set = node_width.GetMax()
        #     node_width.SetValue(width_to_set)
        #     print('Width set to %i' % node_width.GetValue())
        # else:
        #     print('Width not available')
        #
        # node_height = PySpin.CIntegerPtr(nodemap_tldevice.GetNode('Width'))
        # node_height.SetValue(600)
        # if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
        #     height_to_set = node_height.GetMax()
        #     node_height.SetValue(height_to_set)
        #     print('Height set to %i' % node_height.GetValue())
        # else:
        #     print('Height not available')

        return True

    def print_device_info(self, nodemap, cam_num):

        print('Printing device information for camera %d... \n' % cam_num)

        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(), node_feature.ToString() if PySpin.IsReadable(
                                          node_feature) else 'Node not readable'))
            else:
                print('Device control information not available.')
            print()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def configure_device_events(self, cam, cam_id, subscriber):
        """
        This function configures the example to execute device events by enabling all
        types of device events, and then creating and registering a device event handler that
        only concerns itself with an end of exposure event.

        :param INodeMap nodemap: Device nodemap.
        :param CameraPtr cam: Pointer to camera.
        :returns: tuple (result, device_event_handler)
            WHERE
            result is True if successful, False otherwise
            device_event_handler is the event handler
        :rtype: (bool, DeviceEventHandler)
        """
        print('\n*** CONFIGURING DEVICE EVENTS ***\n')

        try:
            success = True

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            #  Retrieve device event selector
            #
            #  *** NOTES ***
            #  Each type of device event must be enabled individually. This is done
            #  by retrieving "EventSelector" (an enumeration node) and then enabling
            #  the device event on "EventNotification" (another enumeration node).
            #
            #  This example only deals with exposure end events. However, instead of
            #  only enabling exposure end events with a simpler device event function,
            #  all device events are enabled while the device event handler deals with
            #  ensuring that only exposure end events are considered. A more standard
            #  use-case might be to enable only the events of interest.
            node_event_selector = PySpin.CEnumerationPtr(nodemap.GetNode('EventSelector'))
            if not PySpin.IsAvailable(node_event_selector) or not PySpin.IsReadable(node_event_selector):
                print('Unable to retrieve event selector entries. Aborting...')
                return False

            # TODO: Only listen for Exposure End Event
            entries = node_event_selector.GetEntries()
            print('Enabling event selector entries...')

            # Enable device events
            #
            # *** NOTES ***
            # In order to enable a device event, the event selector and event
            # notification nodes (both of type enumeration) must work in unison.
            # The desired event must first be selected on the event selector node
            # and then enabled on the event notification node.
            for entry in entries:

                # Select entry on selector node
                node_entry = PySpin.CEnumEntryPtr(entry)
                if not PySpin.IsAvailable(node_entry) or not PySpin.IsReadable(node_entry):
                    # Skip if node fails
                    success = False
                    continue

                node_event_selector.SetIntValue(node_entry.GetValue())

                # Retrieve event notification node (an enumeration node)
                node_event_notification = PySpin.CEnumerationPtr(nodemap.GetNode('EventNotification'))
                if not PySpin.IsAvailable(node_event_notification) or not PySpin.IsWritable(node_event_notification):
                    # Skip if node fails
                    success = False
                    continue

                # Retrieve entry node to enable device event
                node_event_notification_on = PySpin.CEnumEntryPtr(node_event_notification.GetEntryByName('On'))
                if not PySpin.IsAvailable(node_event_notification_on) or not PySpin.IsReadable(
                        node_event_notification_on):
                    # Skip if node fails
                    success = False
                    continue

                node_event_notification.SetIntValue(node_event_notification_on.GetValue())

                print('\t%s: enabled...' % node_entry.GetDisplayName())

            # Create device event handler
            #
            # *** NOTES ***
            # The class has been designed to take in the name of an event. If all
            # events are registered generically, all event types will trigger a
            # device event; on the other hand, if an event handler is registered
            # specifically, only that event will trigger an event.
            device_event_handler = DeviceEventHandler('EventExposureEnd', cam_id, cam, subscriber)

            # Register device event handler
            #
            # *** NOTES ***
            # Device event handlers are registered to cameras. If there are multiple
            # cameras, each camera must have any device event handlers registered to it
            # separately. Note that multiple device event handlers may be registered to a
            # single camera.
            #
            # *** LATER ***
            # Device event handlers must be unregistered manually. This must be done prior
            # to releasing the system and while the device event handlers are still in
            # scope.

            cam.RegisterEventHandler(device_event_handler, 'EventExposureEnd')
            print('Device event handler registered specifically to EventExposureEnd events')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            success = False

        return success, device_event_handler

    def end_camera_capture(self):
        print('End Camera Capture')
        for i, cam in enumerate(self.cam_list):
            cam.EndAcquisition()  # End acquisition

            cam.UnregisterEventHandler(self.device_event_handlers[i])
            print('Device event handler unregistered')

            # Deinitialize camera. Each camera needs to be deinitialized once all images have been acquired.
            cam.DeInit()

        # Release reference to camera. The usage of del is preferred to assigning the variable to None.
        del cam

        self.cam_list.Clear()

        # Release system instance
        self.system.ReleaseInstance()


class CameraTester:

    frames = []
    camera_serial_numbers = []

    windows_initialized = False

    def __init__(self):
        global DEBUG_MODE
        global EXTRACT_PROJECTION_AREA
        DEBUG_MODE = True
        EXTRACT_PROJECTION_AREA = True

        # If this script is started as main, the debug mode is activated by default:
        cam_exposure = EXPOSURE_TIME_MICROSECONDS
        framerate = FRAMERATE
        # cam_exposure = 10000

        thread = threading.Thread(target=self.debug_mode_thread)
        thread.start()
        # thread.join()

        if CALIBRATION_MODE:
            DEBUG_MODE = False  # No Debug Mode wanted in Calibration mode
            cam_exposure = 100000  # Increase Brightness to better see the corners
            self.surface_selector = SurfaceSelector()

        flir_blackfly_s = FlirBlackflyS(cam_exposure=cam_exposure, framerate=framerate, subscriber=self)

        # TODO: Improve
        time.sleep(100000)  # Wait before it stops
        print('ENDING CAMERA TESTING')
        flir_blackfly_s.end_camera_capture()

    def debug_mode_thread(self):
        while True:

            extracted_frames = []

            if not self.windows_initialized:
                if len(self.camera_serial_numbers) > 0:
                    for camera_serial_number in self.camera_serial_numbers:
                        cv2.namedWindow('Flir Blackfly S {}'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), FRAME_WIDTH, FRAME_HEIGHT)

                        if EXTRACT_PROJECTION_AREA:
                            cv2.namedWindow('Flir Camera {} Extracted'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), FRAME_WIDTH, FRAME_HEIGHT)

                            # cv2.namedWindow('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN)
                            # cv2.setWindowProperty('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN,
                            #                       cv2.WINDOW_FULLSCREEN)
                    self.windows_initialized = True
                else:
                    continue

            if len(self.frames) > 0:
                for i, frame in enumerate(self.frames):

                    window_name = 'Flir Blackfly S {}'.format(self.camera_serial_numbers[i])
                    window_name_extracted = 'Flir Camera {} Extracted'.format(self.camera_serial_numbers[i])

                    if DEBUG_MODE:
                        cv2.imshow(window_name, frame)

                    if EXTRACT_PROJECTION_AREA:
                        global table_extractor
                        extracted_frame = table_extractor.extract_table_area(frame, window_name)
                        extracted_frame = cv2.resize(extracted_frame, (3840, 2160))
                        extracted_frames.append(extracted_frame)
                        cv2.imshow(window_name_extracted, extracted_frame)

                    if CALIBRATION_MODE:
                            calibration_finished = self.surface_selector.select_surface(frame, window_name)

                            if calibration_finished:
                                print('[Surface Selector Node]: Calibration finished for {}'.format(window_name))

                self.frames = []

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                sys.exit(0)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(self.camera_serial_numbers) == 0:
            self.camera_serial_numbers = camera_serial_numbers

        self.frames = frames


if __name__ == '__main__':
    CameraTester()




