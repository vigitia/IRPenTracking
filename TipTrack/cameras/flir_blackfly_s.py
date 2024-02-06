import datetime
import os
import time
import cv2

import PySpin

from TipTrack.utility.surface_extractor import SurfaceExtractor
from TipTrack.utility.camera_undistortion_utility import CameraUndistortionUtility


# CONSTANTS and Camera Settings

DEBUG_MODE = False

INTRINSIC_CAMERA_CALIBRATION = False  # Enable to load stored intrinsic camera calibration data and apply to frames.

FRAME_WIDTH = 1920  # 800  # 1920
FRAME_HEIGHT = 1200  # 600  # 1200
EXPOSURE_TIME_MICROSECONDS = 600  # 600  # μs -> must be lower than the frame time (FRAMERATE / 1000)
GAIN = 18  # Controls the amplification of the video signal in dB.
FRAMERATE = 158  # Target number of Frames per Second (Min: 1, Max: 158)
NUM_BUFFERS = 1  # Number of roi buffers per camera

# Specify the Serial Numbers of the primary and secondary camera. Needed for the hardware trigger
SERIAL_NUMBER_MASTER = str(22260470)
SERIAL_NUMBER_SLAVE = str(22260466)


# Global variables

cam_image_master = None
cam_image_slave = None
all_cameras_ready = False
matrices = [[]]


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """ This class defines a custom Event Handler for camera events.

        We want to listen here for the 'EventExposureEnd' in particular.
    """

    start_time = datetime.datetime.now()

    def __init__(self, cam_id, cam, subscriber, rectify_maps=None):
        super(DeviceEventHandler, self).__init__()
        self.cam_id = cam_id
        self.cam = cam
        self.subscriber = subscriber
        self.rectify_maps = rectify_maps

    def OnDeviceEvent(self, event_name):
        """ OnDeviceEvent

            Callback function when a device event occurs.

            Note event_name is a wrapped gcstring, not a Python string, but basic operations such as printing and
            comparing with Python strings are supported.
        """

        # Check if all cameras are already initialized
        global all_cameras_ready
        if not all_cameras_ready:
            # Not all cameras initialized yet, therefore skip this event
            return

        # Print information on specified device event
        # print('\tDevice Event "{}" ({}) from camera {}; {}'.format(event_name, self.GetDeviceEventId(), self.camera_serial_number,
        #                                                            self.count))

        # TODO: How long to wait here? -> Define grabTimeout
        image_result = self.cam.GetNextImage(100)

        # print(image_result.GetTimeStamp() / 1e9)

        if image_result.IsIncomplete():  # Ensure roi completion
            print('[Flir BlackFly S]: Warning: Image incomplete with Image Status %d' % image_result.GetImageStatus())
        else:
            if self.cam_id == SERIAL_NUMBER_MASTER:
                global cam_image_master

                if INTRINSIC_CAMERA_CALIBRATION:
                    cam_image_master = cv2.remap(image_result.GetNDArray(), self.rectify_maps[0], self.rectify_maps[1],
                                                 interpolation=cv2.INTER_LINEAR)
                else:
                    cam_image_master = image_result.GetNDArray()
            elif self.cam_id == SERIAL_NUMBER_SLAVE:
                global cam_image_slave

                if INTRINSIC_CAMERA_CALIBRATION:
                    cam_image_slave = cv2.remap(image_result.GetNDArray(), self.rectify_maps[0], self.rectify_maps[1],
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    cam_image_slave = image_result.GetNDArray()

                # TODO: This is not working properly. Sometimes not both frames are available when the slave camera is finished. This causes dropped frames right now.
                self.check_both_frames_available()

        #  Images retrieved directly from the camera need to be released in order to keep from filling the buffer.
        image_result.Release()

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

            if DEBUG_MODE:
                end_time = datetime.datetime.now()
                run_time = (end_time - self.start_time).microseconds / 1000.0
                expected_time_between_frames_ms = round(1000 / FRAMERATE, 2)
                if run_time > expected_time_between_frames_ms * 1.5:
                    print('[Flir BlackFly S]: WARNING: Time to get both frames was {}ms, but should be {}ms'.format(run_time, expected_time_between_frames_ms))
                self.start_time = datetime.datetime.now()
        elif DEBUG_MODE:
            print('[Flir BlackFly S]: Warning! Not both frames available! Master available: {}, Slave available: {}'.format(cam_image_master is not None, cam_image_slave is not None))


class FlirBlackflyS:

    # Data for optional intrinsic camera calibration
    rectify_maps = {}

    pyspin_system = None  # PySpin System object needed to access the cameras
    cam_list = []  # List containing reference to all cameras

    device_event_handlers = []
    device_serial_numbers = []

    def __init__(self, cam_exposure=EXPOSURE_TIME_MICROSECONDS, subscriber=None, gain=GAIN):

        # Overwrite the default exposure time if needed
        global EXPOSURE_TIME_MICROSECONDS
        EXPOSURE_TIME_MICROSECONDS = cam_exposure

        global GAIN
        GAIN = gain

        self.surface_extractor = SurfaceExtractor()
        self.camera_undistortion_utility = CameraUndistortionUtility(FRAME_WIDTH, FRAME_HEIGHT)

        self.init_cameras(subscriber)

    def get_exposure_time(self):
        return EXPOSURE_TIME_MICROSECONDS

    def get_gain(self):
        return GAIN

    def init_cameras(self, subscriber):
        self.pyspin_system = PySpin.System.GetInstance()  # Retrieve singleton reference to pyspin_system object

        if DEBUG_MODE:
            version = self.pyspin_system.GetLibraryVersion()  # Get current library version
            print('[Flir BlackFly S]: Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        self.cam_list = self.pyspin_system.GetCameras()  # Retrieve list of cameras from the pyspin_system

        num_cameras = self.cam_list.GetSize()
        print('[Flir BlackFly S]: Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            self.cam_list.Clear()  # Clear camera list before releasing pyspin_system
            self.pyspin_system.ReleaseInstance()  # Release pyspin_system instance
            print('[Flir BlackFly S]: No cameras found')
            return

        try:
            if DEBUG_MODE:
                # Retrieve transport layer nodemaps and print device information for each camera
                for i, cam in enumerate(self.cam_list):
                    nodemap_tldevice = cam.GetTLDeviceNodeMap()  # Retrieve Transport layer device nodemap
                    self.print_device_info(nodemap_tldevice, i)  # Print device information

            # Initialize each camera
            for i, cam in enumerate(self.cam_list):
                self.__initialize_camera(cam, i, subscriber)

            cam_slave = self.cam_list.GetBySerial(SERIAL_NUMBER_SLAVE)
            cam_master = self.cam_list.GetBySerial(SERIAL_NUMBER_MASTER)

            cam_slave.BeginAcquisition()  # Begin acquiring images
            cam_master.BeginAcquisition()  # Begin acquiring images

            global all_cameras_ready
            all_cameras_ready = True

        except PySpin.SpinnakerException as ex:
            print('[Flir BlackFly S]: SpinnakerException: %s' % ex)

    def __initialize_camera(self, cam, i, subscriber):
        # Retrieve device serial numbers
        device_serial_number = -1

        node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('[Flir BlackFly S]: Camera %d Serial Number: %s' % (i, device_serial_number))

            if INTRINSIC_CAMERA_CALIBRATION:
                maps = self.camera_undistortion_utility.get_camera_undistort_rectify_maps(
                    'Flir Blackfly S {}.yml'.format(device_serial_number))
                self.rectify_maps[device_serial_number] = maps

        self.device_serial_numbers.append(device_serial_number)

        global matrices
        if device_serial_number == SERIAL_NUMBER_MASTER:
            # Make sure that the master camera will always be the first in the list
            matrices[0] = self.surface_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT,
                                                                'Flir Blackfly S {}'.format(device_serial_number))
        else:
            matrices.append(self.surface_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT,
                                                                  'Flir Blackfly S {}'.format(
                                                                      device_serial_number)))

        cam.Init()  # Initialize camera

        self.init_hardware_trigger(cam, device_serial_number)

        device_event_handler = self.configure_device_events(cam, device_serial_number, subscriber)
        self.device_event_handlers.append(device_event_handler)

        success = self.apply_camera_settings(cam, device_serial_number)
        if not success:
            print('[Flir BlackFly S]: Errors while applying settings to the cameras')
            time.sleep(10)

    def init_hardware_trigger(self, cam, device_serial_number):
        """ init_hardware_trigger

            This will set up the cameras in a way so that the master camera will control when all other cameras
            take a picture. This should sync the images as good as possible.
        """
        if device_serial_number == SERIAL_NUMBER_MASTER:  # Set Master
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            cam.LineMode.SetValue(PySpin.LineMode_Output)
            cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
            cam.V3_3Enable.SetValue(True)
        elif device_serial_number == SERIAL_NUMBER_SLAVE:  # Set Slave
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            # Set TriggerActivation Falling Edge TODO: better Rising Edge? ...
            cam.TriggerActivation.SetValue(PySpin.TriggerActivation_FallingEdge)
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        else:
            print('[Flir BlackFly S]: WRONG CAMERA ID FOR MASTER AND SLAVE!:', type(device_serial_number))

    def apply_camera_settings(self, cam, serial_number):

        # Set Acquisition Mode to Continuous: acquires images continuously
        if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        else:
            print("[Flir BlackFly S]: Error setting AcquisionMode: no access")
            return False

        # Set AutoExposure off
        if cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        else:
            print('[Flir BlackFly S]: PySpin:Camera:Failed to turn off Autoexposure')
            return False

        # Check if AutoExposure is off and exposure mode is set to "Timed" (needed for manually setting exposure)
        if cam.ExposureMode.GetValue() != PySpin.ExposureMode_Timed:
            print('[Flir BlackFly S]: PySpin:Camera: Can not set exposure! Exposure Mode needs to be Timed')
            return False
        if cam.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
            print('[Flir BlackFly S]: PySpin:Camera: Can not set exposure! Exposure is Auto')
            return False

        # Set Exposure Time (in microseconds).
        # Exposure time should not be greater than frame time, this would reduce the resulting framerare
        if cam.ExposureTime.GetAccessMode() == PySpin.RW:
            cam.ExposureTime.SetValue(
                max(cam.ExposureTime.GetMin(), min(cam.ExposureTime.GetMax(), float(EXPOSURE_TIME_MICROSECONDS))))
        else:
            print('[Flir BlackFly S]: PySpin:Camera:Failed to set exposure to:{}'.format(EXPOSURE_TIME_MICROSECONDS))
            return False

        # Set GainAuto to off.
        if cam.GainAuto.GetAccessMode() == PySpin.RW:
            cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        else:
            print('[Flir BlackFly S]: PySpin:Camera:Failed to set GainAuto to off')
            return False

        # Set Gain. Controls the amplification of the video signal in dB.
        if cam.Gain.GetAccessMode() == PySpin.RW:
            cam.Gain.SetValue(
                max(cam.Gain.GetMin(), min(cam.Gain.GetMax(), float(GAIN))))
        else:
            print("[Flir BlackFly S]: PySpin:Camera:Failed to set Gain to:{}".format(GAIN))
            return False

        # TODO: Check if manually setting the Black Level has any impact on our output frames

        # Set Acquisiton Frame Rate only for Master camera
        if serial_number == SERIAL_NUMBER_MASTER:
            # Set Acquisiton Frame Rate Enable = True to be able to manually set the framerate
            if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRateEnable.SetValue(True)
            else:
                print("[Flir BlackFly S]: PySpin:Camera:AcquisionFrameRateEnable: no access")
                return False

            # Set Camera Acquisition Framerate
            if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRate.SetValue(min(cam.AcquisitionFrameRate.GetMax(), FRAMERATE))
            else:
                print("[Flir BlackFly S]: PySpin:Camera:Failed to set CameraAcquisitionFramerate to:{}".format(FRAMERATE))
                return False

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('[Flir BlackFly S]: Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            print('[Flir BlackFly S]: Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('[Flir BlackFly S]: Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(
                stream_buffer_count_mode_manual):
            print('[Flir BlackFly S]: Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('[Flir BlackFly S]: Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        # Display Buffer Info
        # print('Default Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        # print('Default Buffer Count: %d' % buffer_count.GetValue())
        # print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        buffer_count.SetValue(NUM_BUFFERS)

        # handling_mode_entry = handling_mode.GetEntryByName('NewestFirst')
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
        # handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')

        handling_mode.SetIntValue(handling_mode_entry.GetValue())

        # TODO: Also set WIDTH AND HEIGHT, X_OFFSET, Y_OFFSET

        # if camera.Width.GetAccessMode() == PySpin.RW:
        #     camera.Width.SetValue(1920)
        #     print("PySpin:Camera:Width:{}".format(camera.Width.GetValue()))
        # else:
        #     print("PySpin:Camera:Failed to set Width to:{}".format(1920))
        #     return False
        #
        # if camera.Height.GetAccessMode() == PySpin.RW:
        #     camera.Height.SetValue(1200)
        #     print("PySpin:Camera:Height:{}".format(camera.Height.GetValue()))
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

        if DEBUG_MODE:
            print("[Flir BlackFly S]: AcquistionMode: {}".format(cam.AcquisitionMode.GetValue()))
            print('[Flir BlackFly S]: ExposureAuto: {}'.format(cam.ExposureAuto.GetValue()))
            print('[Flir BlackFly S]: PySpin:Camera:Exposure:{}'.format(cam.ExposureTime.GetValue()))
            print('[Flir BlackFly S]: PySpin:Camera:GainAuto: {}'.format(cam.GainAuto.GetValue()))
            print("[Flir BlackFly S]: PySpin:Camera:Gain:{}".format(cam.Gain.GetValue()))
            print("[Flir BlackFly S]: PySpin:Camera:AcquisionFrameRateEnable: {}".format(
                cam.AcquisitionFrameRateEnable.GetValue()))
            print('[Flir BlackFly S]: PySpin:Camera:CameraAcquisitionFramerate:', cam.AcquisitionFrameRate.GetValue())
            print('[Flir BlackFly S]: PySpin:Camera:StreamBufferCountMode: Manual')
            print('[Flir BlackFly S]: PySpin:Camera:BufferCount: %d' % buffer_count.GetValue())
            print('[Flir BlackFly S]: PySpin:Camera:BufferHandlingMode: %s' % handling_mode_entry.GetDisplayName())
            print('------------------------------------------------------------------')

        return True

    def print_device_info(self, nodemap, cam_num):

        print('[Flir BlackFly S]: Printing device information for camera %d... \n' % cam_num)

        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('[Flir BlackFly S]: %s: %s' % (node_feature.GetName(), node_feature.ToString() if PySpin.IsReadable(
                                          node_feature) else 'Node not readable'))
            else:
                print('[Flir BlackFly S]: Device control information not available.')

        except PySpin.SpinnakerException as ex:
            print('[Flir BlackFly S]: Error in print_device_info(): %s' % ex)
            return False

        return result

    def configure_device_events(self, camera, camera_serial_number, subscriber):
        """ configure_device_events

            This function configures the example to execute device events by enabling all types of device events,
            and then creating and registering a device event handler that only concerns itself with an end of exposure
            event.
        """

        try:
            success = True

            # Retrieve GenICam nodemap
            nodemap = camera.GetNodeMap()

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
                print('[Flir BlackFly S]: Unable to retrieve event selector entries. Aborting...')
                return False

            # TODO: Only listen for Exposure End Event
            entries = node_event_selector.GetEntries()
            # print('Enabling event selector entries')

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

                # print('%s: enabled' % node_entry.GetDisplayName())

            # Create device event handler
            #
            # *** NOTES ***
            # The class has been designed to take in the name of an event. If all
            # events are registered generically, all event types will trigger a
            # device event; on the other hand, if an event handler is registered
            # specifically, only that event will trigger an event.
            if INTRINSIC_CAMERA_CALIBRATION:
                device_event_handler = DeviceEventHandler(camera_serial_number, camera, subscriber,
                                                          self.rectify_maps[camera_serial_number])
            else:
                device_event_handler = DeviceEventHandler(camera_serial_number, camera, subscriber)

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
            # to releasing the pyspin_system and while the device event handlers are still in
            # scope.

            camera.RegisterEventHandler(device_event_handler, 'EventExposureEnd')
            # print('Device event handler registered specifically to EventExposureEnd events')
            # print('')

        except PySpin.SpinnakerException as ex:
            print('Error in configure_device_events(): %s' % ex)
            success = False

        if not success:
            print('[Flir BlackFly S]: Error in configure_device_events()')
            time.sleep(10)

        return device_event_handler

    def end_camera_capture(self):
        print('[Flir BlackFly S]: End Camera Capture')
        for i, cam in enumerate(self.cam_list):
            cam.EndAcquisition()  # End acquisition

            cam.UnregisterEventHandler(self.device_event_handlers[i])
            print('[Flir BlackFly S]: Device event handler unregistered')

            # Deinitialize camera. Each camera needs to be deinitialized once all images have been acquired.
            cam.DeInit()

        # Release reference to camera. The usage of del is preferred to assigning the variable to None.
        del cam

        self.cam_list.Clear()

        # Release pyspin_system instance
        # TODO: This causes error "_PySpin.SpinnakerException: Spinnaker: Can't clear a camera because something still
        #  holds a reference to the camera [-1004]"
        # self.pyspin_system.ReleaseInstance()




