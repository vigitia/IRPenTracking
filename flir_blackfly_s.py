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

DEBUG_MODE = False
EXTRACT_PROJECTION_AREA = False
SHOW_DEBUG_STACKED_FRAMES = False

CALIBRATION_DATA_PATH = ''

CALIBRATION_MODE = False

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1200

CAM_EXPOSURE = 300
FRAMERATE = 158


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return retval
        return wrapper
    return timeit_decorator


class EventType:
    """
    'Enum' for choosing whether to register a event specifically for exposure end events
    or universally for all events.
    """
    GENERIC = 0
    SPECIFIC = 1

CHOSEN_EVENT = EventType.SPECIFIC


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """
    This class defines the properties, parameters, and the event handler itself. Take a
    moment to notice what parts of the class are mandatory, and what have been
    added for demonstration purposes. First, any class used to define device
    events must inherit from DeviceEventHandler. Second, the method signature of
    OnDeviceEvent() must also be consistent. Everything else - including the
    constructor, destructor, properties, and body of OnDeviceEvent() - are
    particular to the example.
    """
    def __init__(self, eventname, cam_id):
        """
        This constructor registers an event name to be used on device events.

        :param eventname: Name of event to register.
        :type eventname: str
        :rtype: None
        """
        super(DeviceEventHandler, self).__init__()
        self.event_name = eventname
        self.count = 0
        self.cam_id = cam_id

    def OnDeviceEvent(self, eventname):
        """
        Callback function when a device event occurs.
        Note eventname is a wrapped gcstring, not a Python string, but basic operations such as printing and comparing
        with Python strings are supported.

        :param eventname: gcstring representing the name of the occurred event.
        :type eventname: gcstring
        :rtype: None
        """
        if eventname == self.event_name:
            self.count += 1

            # Print information on specified device event
            print('\tDevice Event "{}" ({}) from camera {}; {}'.format(eventname, self.GetDeviceEventId(), self.cam_id,
                                                                     self.count))
            # print('\tDevice event %s (%i) from camera number %i...' % (eventname,
            #                                                    self.GetDeviceEventId(),
            #                                                    self.count))
        else:
            # Print no information on non-specified event
            print('\tDevice event occurred; not %s; ignoring...' % self.event_name)


class FlirBlackflyS:

    camera_matrices = []
    dist_matrices = []
    rectify_maps = []

    system = None
    cam_list = []

    device_event_handlers = []

    newest_frames = []
    matrices = []
    new_frames_available = False

    start_time = time.time()
    frame_counter = 0

    def __init__(self, cam_exposure=CAM_EXPOSURE, framerate=FRAMERATE):
        global CAM_EXPOSURE
        global FRAMERATE
        CAM_EXPOSURE = cam_exposure
        FRAMERATE = framerate

        self.surface_selector = SurfaceSelector()
        self.table_extractor = TableExtractionService()

        self.started = False
        self.read_lock = threading.Lock()

        self.init_cameras()

    def start(self):
        if self.started:
            return None
        else:
            self.started = True
            self.thread = threading.Thread(target=self.update, args=())
            # thread.daemon = True
            self.thread.start()
            return self

    def update(self):
        while self.started:
            self.process_frames()

            if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
                if DEBUG_MODE:
                    print("FPS: %s (Warning: DEBUG_MODE might reduce FPS)" % round(self.frame_counter / (time.time() - self.start_time), 1))
                else:
                    print("FPS: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
                self.frame_counter = 0
                self.start_time = time.time()

        for i, cam in enumerate(self.cam_list):
            cam.EndAcquisition()  # End acquisition

            cam.UnregisterEventHandler(self.device_event_handlers[i])
            print('Device event handler unregistered...\n')

            # Deinitialize camera. Each camera needs to be deinitialized once all images have been acquired.
            cam.DeInit()

        # Release reference to camera. The usage of del is preferred to assigning the variable to None.
        del cam

        self.cam_list.Clear()

        # Release system instance
        self.system.ReleaseInstance()

    # @timeit('Flir Blackfly S')
    def process_frames(self):

        newest_frames = []

        self.frame_counter += 1

        if len(self.matrices) == 0:
            for i, cam in enumerate(self.cam_list):
                self.matrices.append(self.table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Camera {}'.format(i)))

        for cam in self.cam_list:
            try:
                #start = datetime.datetime.now()
                image_result = cam.GetNextImage(1000)  # grabTimeout=1000

                if image_result.IsIncomplete():  # Ensure image completion
                    print('Image incomplete with image status %d' % image_result.GetImageStatus())
                else:
                    image_data = image_result.GetNDArray()
                    newest_frames.append(image_data)

                    # Convert image to mono 8
                    # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                #  Images retrieved directly from the camera need to be released in order to keep from filling the
                #  buffer.
                image_result.Release()  # Release image
                #end = datetime.datetime.now()
                #print('GetNextImage', (end - start).microseconds / 1000.0)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)

        for i, frame in enumerate(newest_frames):
            if self.camera_matrices[i] is not None and self.dist_matrices[i] is not None:
                newest_frames[i] = cv2.remap(frame, self.rectify_maps[i][0], self.rectify_maps[i][1],
                                             interpolation=cv2.INTER_LINEAR)

                # newest_frames[i] = cv2.undistort(frame, self.camera_matrices[i], self.dist_matrices[i], None, None)

        # with self.read_lock:
        #     self.newest_frames = newest_frames
        #     self.new_frames_available = True

        if DEBUG_MODE:
            extracted_frames = []

            for i, frame in enumerate(newest_frames):
                cv2.imshow('Flir Camera {}'.format(i), frame)

                if EXTRACT_PROJECTION_AREA:
                    extracted_frame = self.table_extractor.extract_table_area(frame, 'Flir Camera {}'.format(i))
                    extracted_frames.append(extracted_frame)
                    cv2.imshow('Flir Camera {} extracted'.format(i), extracted_frame)

            if SHOW_DEBUG_STACKED_FRAMES and len(extracted_frames) == 2:
                zeroes = np.zeros(extracted_frames[0].shape, 'uint8')
                fake_color = np.dstack((extracted_frames[0], extracted_frames[1], zeroes))

                cv2.namedWindow('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Flir Camera Frames combined', fake_color)

        if CALIBRATION_MODE:
            for i, cam in enumerate(self.cam_list):

                calibration_finished = self.surface_selector.select_surface(newest_frames[i], 'Flir Camera {}'.format(i))

                if calibration_finished:
                    print("[Surface Selector Node]: Calibration Finished for camera {}".format(i))

        if DEBUG_MODE or CALIBRATION_MODE:
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.started = False
                cv2.destroyAllWindows()
                sys.exit(0)

    def init_cameras(self):
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()
        print('Number of cameras detected: %d' % num_cameras)

        self.load_camera_calibration_data(num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            self.cam_list.Clear()  # Clear camera list before releasing system
            self.system.ReleaseInstance()  # Release system instance

            print('No camera found')
            return False

        try:
            if DEBUG_MODE:
                # Retrieve transport layer nodemaps and print device information for each camera
                for i, cam in enumerate(self.cam_list):
                    nodemap_tldevice = cam.GetTLDeviceNodeMap()  # Retrieve Transport layer device nodemap
                    self.print_device_info(nodemap_tldevice, i)  # Print device information

            # Initialize each camera
            for i, cam in enumerate(self.cam_list):

                device_serial_number = -1

                # Retrieve device serial number for filename
                node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
                if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                    device_serial_number = node_device_serial_number.GetValue()
                    print('Camera %d serial number: %s' % (i, device_serial_number))

                cam.Init()  # Initialize camera

                success, device_event_handler = self.configure_device_events(cam, device_serial_number)
                if not success:
                    print('error in configure_device_events()')
                self.device_event_handlers.append(device_event_handler)

                self.apply_camera_settings(cam)

            for i, cam in enumerate(self.cam_list):
                cam.BeginAcquisition()  # Begin acquiring images
                print('Camera %d started acquiring images\n' % i)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

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
            except:
                print('Cant load calibration data for FlirBlackflyS {}.yml'.format(i))

    def apply_camera_settings(self, cam):

        # TODO: Also set GAIN, WIDTH AND HEIGHT, X_OFFSET, Y_OFFSET

        # Set Acquisition Mode to Continuous
        if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            print("PySpin:Camera:AcquistionMode: {}".format(cam.AcquisitionMode.GetValue()))
        else:
            print("PySpin:Camera:AcquisionMode: no access")

        # Set Autoexposure off
        if cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            print("PySpin:Camera:ExposureAuto: {}".format(cam.ExposureAuto.GetValue()))
        else:
            print("PySpin:Camera:Failed to set Autoexposure to off")

        # Check if Autoexposure is off and exposure mode is set to "Timed" (needed for manually setting exposure)
        if cam.ExposureMode.GetValue() != PySpin.ExposureMode_Timed:
            print("PySpin:Camera:Can not set exposure! Exposure Mode needs to be Timed")
        if cam.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
            print("PySpin:Camera:Can not set exposure! Exposure is Auto")

        # Set Exposure Time
        if cam.ExposureTime.GetAccessMode() == PySpin.RW:
            cam.ExposureTime.SetValue(
                max(cam.ExposureTime.GetMin(), min(cam.ExposureTime.GetMax(), float(CAM_EXPOSURE))))
            print("PySpin:Camera:Exposure:{}".format(cam.ExposureTime.GetValue()))
        else:
            print("PySpin:Camera:Failed to set expsosure to:{}".format(CAM_EXPOSURE))

        # Set Acquisiton Frame Rate Enable = True
        if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
            cam.AcquisitionFrameRateEnable.SetValue(True)
            print("PySpin:Camera:AcquisionFrameRateEnable: {}".format(cam.AcquisitionFrameRateEnable.GetValue()))
        else:
            print("PySpin:Camera:AcquisionFrameRateEnable: no access")

        # Set Camera Acquisition Framerate
        if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
            cam.AcquisitionFrameRate.SetValue(min(cam.AcquisitionFrameRate.GetMax(), FRAMERATE))
            print('PySpin:Camera:CameraAcquisitionFramerate:', cam.AcquisitionFrameRate.GetValue())
        else:
            print("PySpin:Camera:Failed to set CameraAcquisitionFramerate to:{}".format(FRAMERATE))

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

        NUM_BUFFERS = 1

        buffer_count.SetValue(NUM_BUFFERS)

        print('Buffer count set to: %d' % buffer_count.GetValue())

        # handling_mode_entry = handling_mode.GetEntryByName('NewestFirst')
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')

        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        print('Buffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())

        # node_width = PySpin.CIntegerPtr(nodemap_tldevice.GetNode('Width'))
        # node_width.SetValue(800)
        # if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
        #     width_to_set = node_width.GetMax()
        #     node_width.SetValue(width_to_set)
        #     print('Width set to %i' % node_width.GetValue())
        # else:
        #     print('Width not available')

        # # Set acquisition mode to continuous
        # node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
        # if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        #     print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
        #     return False
        #
        # node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        # if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
        #         node_acquisition_mode_continuous):
        #     print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
        #             Aborting... \n' % i)
        #     return False
        #
        # acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        #
        # node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        #
        # print('Camera %d acquisition mode set to continuous' % i)

    # def set_camera_buffer(self):


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

    def get_camera_frames(self):

        if self.new_frames_available:
            # self.frame_counter += 1
            with self.read_lock:

                self.new_frames_available = False

                # if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
                #     print("FPS1: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
                #     self.frame_counter = 0
                #     self.start_time = time.time()

                return self.newest_frames, self.matrices
        else:
            # if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
            #     print("FPS2: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
            #     self.frame_counter = 0
            #     self.start_time = time.time()

            return [], []

    def configure_device_events(self, cam, cam_id):
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
            device_event_handler = DeviceEventHandler('EventExposureEnd', cam_id)

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
            if CHOSEN_EVENT == EventType.GENERIC:

                # Device event handlers registered generally will be triggered by any device events.
                cam.RegisterEventHandler(device_event_handler)

                print('Device event handler registered generally...')

            elif CHOSEN_EVENT == EventType.SPECIFIC:

                # Device event handlers registered to a specified event will only
                # be triggered by the type of event is it registered to.
                cam.RegisterEventHandler(device_event_handler, 'EventExposureEnd')

                print('Device event handler registered specifically to EventExposureEnd events...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            success = False

        return success, device_event_handler


if __name__ == '__main__':
    # If this script is started as main, the debug mode is activated by default:
    DEBUG_MODE = False
    EXTRACT_PROJECTION_AREA = False
    cam_exposure = CAM_EXPOSURE
    framerate = FRAMERATE

    # cam_exposure = 80000

    if CALIBRATION_MODE:
        DEBUG_MODE = False  # No Debug Mode wanted in Calibration mode
        cam_exposure = 100000  # Increase Brightness to better see the corners
    flir_blackfly_s = FlirBlackflyS(cam_exposure=cam_exposure, framerate=framerate)
    flir_blackfly_s.start()
