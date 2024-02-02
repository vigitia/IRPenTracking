import sys
import time

import PySpin

import cv2

FRAME_WIDTH = 1920  # 800  # 1920
FRAME_HEIGHT = 1200  # 600  # 1200
EXPOSURE_TIME_MICROSECONDS = 600  # 600  # μs -> must be lower than the frame time (FRAMERATE / 1000)
GAIN = 18  # Controls the amplification of the video signal in dB.
FRAMERATE = 158  # Target number of Frames per Second (Min: 1, Max: 158)
NUM_BUFFERS = 1  # Number of roi buffers per camera

device_event_handlers = []


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """ This class defines a custom Event Handler for camera events.

        We want to listen here for the 'EventExposureEnd' in particular.
    """

    def __init__(self, cam_id, cam_for_handler):
        super(DeviceEventHandler, self).__init__()
        self.cam_id = cam_id
        self.cam = cam_for_handler
        print('Init DeviceEventHandler')

    def OnDeviceEvent(self, event_name):
        """ OnDeviceEvent

            Callback function when a device event occurs.

            Note event_name is a wrapped gcstring, not a Python string, but basic operations such as printing and
            comparing with Python strings are supported.
        """

        # Print information on specified device event
        #print('\tDevice Event "{}" ({})'.format(event_name, self.GetDeviceEventId()))

        # TODO: How long to wait here? -> Define grabTimeout
        image_result = self.cam.GetNextImage(100)

        # print(image_result.GetTimeStamp() / 1e9)

        if image_result.IsIncomplete():  # Ensure roi completion
            print('[Flir BlackFly S]: Warning: Image incomplete with Image Status %d' % image_result.GetImageStatus())
        else:
            cam_image = image_result.GetNDArray()
            print('Received frame:', cam_image.shape)


        #  Images retrieved directly from the camera need to be released in order to keep from filling the buffer.
        image_result.Release()


def configure_device_events(camera, camera_serial_number):
    """ configure_device_events

        This function configures the example to execute device events by enabling all types of device events,
        and then creating and registering a device event handler that only concerns itself with an end of exposure
        event.
    """

    device_event_handler = None

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

        device_event_handler = DeviceEventHandler(camera_serial_number, camera)

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


def apply_camera_settings(cam_wip):
    # Set Acquisition Mode to Continuous: acquires images continuously
    if cam_wip.AcquisitionMode.GetAccessMode() == PySpin.RW:
        cam_wip.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    else:
        print("[Flir BlackFly S]: Error setting AcquisionMode: no access")
        return False

    # Set AutoExposure off
    if cam_wip.ExposureAuto.GetAccessMode() == PySpin.RW:
        cam_wip.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    else:
        print('[Flir BlackFly S]: PySpin:Camera:Failed to turn off Autoexposure')
        return False

    # Check if AutoExposure is off and exposure mode is set to "Timed" (needed for manually setting exposure)
    if cam_wip.ExposureMode.GetValue() != PySpin.ExposureMode_Timed:
        print('[Flir BlackFly S]: PySpin:Camera: Can not set exposure! Exposure Mode needs to be Timed')
        return False
    if cam_wip.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
        print('[Flir BlackFly S]: PySpin:Camera: Can not set exposure! Exposure is Auto')
        return False

    # Set Exposure Time (in microseconds).
    # Exposure time should not be greater than frame time, this would reduce the resulting framerare
    if cam_wip.ExposureTime.GetAccessMode() == PySpin.RW:
        cam_wip.ExposureTime.SetValue(
            max(cam_wip.ExposureTime.GetMin(), min(cam_wip.ExposureTime.GetMax(), float(EXPOSURE_TIME_MICROSECONDS))))
    else:
        print('[Flir BlackFly S]: PySpin:Camera:Failed to set exposure to:{}'.format(EXPOSURE_TIME_MICROSECONDS))
        return False

    # Set GainAuto to off.
    if cam_wip.GainAuto.GetAccessMode() == PySpin.RW:
        cam_wip.GainAuto.SetValue(PySpin.GainAuto_Off)
    else:
        print('[Flir BlackFly S]: PySpin:Camera:Failed to set GainAuto to off')
        return False

    # Set Gain. Controls the amplification of the video signal in dB.
    if cam_wip.Gain.GetAccessMode() == PySpin.RW:
        cam_wip.Gain.SetValue(
            max(cam_wip.Gain.GetMin(), min(cam_wip.Gain.GetMax(), float(GAIN))))
    else:
        print("[Flir BlackFly S]: PySpin:Camera:Failed to set Gain to:{}".format(GAIN))
        return False

    # TODO: Check if manually setting the Black Level has any impact on our output frames

    # Set Acquisiton Frame Rate Enable = True to be able to manually set the framerate
    if cam_wip.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
        cam_wip.AcquisitionFrameRateEnable.SetValue(True)
    else:
        print("[Flir BlackFly S]: PySpin:Camera:AcquisionFrameRateEnable: no access")
        return False

    # Set Camera Acquisition Framerate
    if cam_wip.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
        cam_wip.AcquisitionFrameRate.SetValue(min(cam_wip.AcquisitionFrameRate.GetMax(), FRAMERATE))
    else:
        print("[Flir BlackFly S]: PySpin:Camera:Failed to set CameraAcquisitionFramerate to:{}".format(FRAMERATE))
        return False

    # Retrieve Stream Parameters device nodemap
    s_node_map = cam_wip.GetTLStreamNodeMap()

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

    print("[Flir BlackFly S]: AcquistionMode: {}".format(cam_wip.AcquisitionMode.GetValue()))
    print('[Flir BlackFly S]: ExposureAuto: {}'.format(cam_wip.ExposureAuto.GetValue()))
    print('[Flir BlackFly S]: PySpin:Camera:Exposure:{}'.format(cam_wip.ExposureTime.GetValue()))
    print('[Flir BlackFly S]: PySpin:Camera:GainAuto: {}'.format(cam_wip.GainAuto.GetValue()))
    print("[Flir BlackFly S]: PySpin:Camera:Gain:{}".format(cam_wip.Gain.GetValue()))
    print("[Flir BlackFly S]: PySpin:Camera:AcquisionFrameRateEnable: {}".format(
        cam_wip.AcquisitionFrameRateEnable.GetValue()))
    print('[Flir BlackFly S]: PySpin:Camera:CameraAcquisitionFramerate:', cam_wip.AcquisitionFrameRate.GetValue())
    print('[Flir BlackFly S]: PySpin:Camera:StreamBufferCountMode: Manual')
    print('[Flir BlackFly S]: PySpin:Camera:BufferCount: %d' % buffer_count.GetValue())
    print('[Flir BlackFly S]: PySpin:Camera:BufferHandlingMode: %s' % handling_mode_entry.GetDisplayName())
    print('------------------------------------------------------------------')

    return True


def __initialize_camera(cam_new, cam_index):
    # Retrieve device serial numbers
    device_serial_number = -1

    node_device_serial_number = PySpin.CStringPtr(cam_new.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
    if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
        device_serial_number = node_device_serial_number.GetValue()
        print('[Flir BlackFly S]: Camera %d Serial Number: %s' % (cam_index, device_serial_number))

    cam_new.Init()  # Initialize camera

    device_event_handler = configure_device_events(cam_new, device_serial_number)
    device_event_handlers.append(device_event_handler)

    success = apply_camera_settings(cam_new)
    if not success:
        print('[Flir BlackFly S]: Errors while applying settings to the cameras')
        sys.exit(1)


def print_device_info(nodemap, cam_num):
    print('[Flir BlackFly S]: Printing device information for camera %d... \n' % cam_num)

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print(
                    '[Flir BlackFly S]: %s: %s' % (node_feature.GetName(), node_feature.ToString() if PySpin.IsReadable(
                        node_feature) else 'Node not readable'))
        else:
            print('[Flir BlackFly S]: Device control information not available.')

    except PySpin.SpinnakerException as spin_exception:
        print('[Flir BlackFly S]: Error in print_device_info(): %s' % spin_exception)
        return False

    return result


pyspin_system = PySpin.System.GetInstance()  # Retrieve singleton reference to pyspin_system object

version = pyspin_system.GetLibraryVersion()  # Get current library version
print('[Flir BlackFly S]: Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

cam_list = pyspin_system.GetCameras()  # Retrieve list of cameras from the pyspin_system

num_cameras = cam_list.GetSize()
print('[Flir BlackFly S]: Number of cameras detected: %d' % num_cameras)

# Finish if there are no cameras
if num_cameras == 0:
    cam_list.Clear()  # Clear camera list before releasing pyspin_system
    pyspin_system.ReleaseInstance()  # Release pyspin_system instance
    print('[Flir BlackFly S]: No cameras found')
else:
    try:

        # Retrieve transport layer nodemaps and print device information for each camera
        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap()  # Retrieve Transport layer device nodemap
            print_device_info(nodemap_tldevice, i)  # Print device information

        # Initialize each camera
        for i, cam in enumerate(cam_list):
            __initialize_camera(cam, i)

            print('Begin Acquisition')
            cam.BeginAcquisition()  # Begin acquiring images

    except PySpin.SpinnakerException as ex:
        print('[Flir BlackFly S]: SpinnakerException: %s' % ex)

    time.sleep(3)

    print('[Flir BlackFly S]: End Camera Capture')
    for i, cam in enumerate(cam_list):
        cam.EndAcquisition()  # End acquisition

        cam.UnregisterEventHandler(device_event_handlers[i])

        # Deinitialize camera. Each camera needs to be deinitialized once all images have been acquired.
        cam.DeInit()

    # Release reference to camera. The usage of del is preferred to assigning the variable to None.
    del cam

    cam_list.Clear()
