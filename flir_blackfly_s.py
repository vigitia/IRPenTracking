#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import threading

import PySpin
import cv2

from surface_selector import SurfaceSelector
from table_extraction_service import TableExtractionService


DEBUG_MODE = False
CALIBRATION_MODE = False

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1200


class FlirBlackflyS:

    system = None
    cam_list = []

    newest_frames = []
    matrices = []
    new_frames_available = False

    def __init__(self):
        self.surface_selector = SurfaceSelector()
        self.table_extractor = TableExtractionService()

        self.init_cameras()

    def start(self):
        print('START!')
        if self.started:
            return None
        else:
            self.started = True
            self.thread = threading.Thread(target=self.update, args=())
            # thread.daemon = True
            self.thread.start()
            return self

    def update(self):
        print('Update')
        while self.started:
            self.process_frames()

        # *** NOTES ***
        # Again, each camera must be deinitialized separately by first
        # selecting the camera and then deinitializing it.
        for cam in self.cam_list:
            # End acquisition
            cam.EndAcquisition()

            # Deinitialize camera
            cam.DeInit()

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

        self.cam_list.Clear()

        # Release system instance
        self.system.ReleaseInstance()

    def process_frames(self):

        newest_frames = []

        if len(self.matrices) == 0:
            for i, cam in enumerate(self.cam_list):
                self.matrices.append(self.table_extractor.get_homography(FRAME_WIDTH, FRAME_HEIGHT, 'Flir Camera {}'.format(i)))

        for i, cam in enumerate(self.cam_list):
            try:
                # Retrieve next received image and ensure image completion
                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()

                    newest_frames.append(image_data)

                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    # print('Camera %d grabbed image, width = %d, height = %d' % (i, width, height))

                    # Convert image to mono 8
                    # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Save image
                    # image_converted.Save(filename)
                    # print('Image saved at %s' % filename)

                    # print(image_data.shape)

                    if DEBUG_MODE:
                        cv2.imshow('Flir Camera {}'.format(i), image_data)


                # Release image
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)

        with self.read_lock:
            self.newest_frames = newest_frames
            self.new_frames_available = True

        if CALIBRATION_MODE:
            # print(self.left_ir_image.shape)

            for i, cam in enumerate(self.cam_list):

                calibration_finished = self.surface_selector.select_surface(newest_frames[i], 'Flir Camera {}'.format(i))

                if calibration_finished:
                    print("[Surface Selector Node]: Calibration Finished for camera {}".format(i))

        if DEBUG_MODE or CALIBRATION_MODE:
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                continue_recording = False
                cv2.destroyAllWindows()
                sys.exit(0)


    def configure_custom_image_settings(self, nodemap):
        """
           Configures a number of settings on the camera including offsets  X and Y, width,
           height, and pixel format. These settings must be applied before BeginAcquisition()
           is called; otherwise, they will be read only. Also, it is important to note that
           settings are applied immediately. This means if you plan to reduce the width and
           move the x offset accordingly, you need to apply such changes in the appropriate order.

           :param nodemap: GenICam nodemap.
           :type nodemap: INodeMap
           :return: True if successful, False otherwise.
           :rtype: bool
           """
        print('\n*** CONFIGURING CUSTOM IMAGE SETTINGS *** \n')

        try:
            result = True

            # Apply mono 8 pixel format
            #
            # *** NOTES ***
            # Enumeration nodes are slightly more complicated to set than other
            # nodes. This is because setting an enumeration node requires working
            # with two nodes instead of the usual one.
            #
            # As such, there are a number of steps to setting an enumeration node:
            # retrieve the enumeration node from the nodemap, retrieve the desired
            # entry node from the enumeration node, retrieve the integer value from
            # the entry node, and set the new value of the enumeration node with
            # the integer value from the entry node.
            #
            # Retrieve the enumeration node from the nodemap
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):

                # Retrieve the desired entry node from the enumeration node
                node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono8'))
                if PySpin.IsAvailable(node_pixel_format_mono8) and PySpin.IsReadable(node_pixel_format_mono8):

                    # Retrieve the integer value from the entry node
                    pixel_format_mono8 = node_pixel_format_mono8.GetValue()

                    # Set integer as new value for enumeration node
                    node_pixel_format.SetIntValue(pixel_format_mono8)

                    print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())

                else:
                    print('Pixel format mono 8 not available...')

            else:
                print('Pixel format not available...')

            # Apply minimum to offset X
            #
            # *** NOTES ***
            # Numeric nodes have both a minimum and maximum. A minimum is retrieved
            # with the method GetMin(). Sometimes it can be important to check
            # minimums to ensure that your desired value is within range.
            node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
            if PySpin.IsAvailable(node_offset_x) and PySpin.IsWritable(node_offset_x):

                node_offset_x.SetValue(node_offset_x.GetMin())
                print('Offset X set to %i...' % node_offset_x.GetMin())

            else:
                print('Offset X not available...')

            # Apply minimum to offset Y
            #
            # *** NOTES ***
            # It is often desirable to check the increment as well. The increment
            # is a number of which a desired value must be a multiple of. Certain
            # nodes, such as those corresponding to offsets X and Y, have an
            # increment of 1, which basically means that any value within range
            # is appropriate. The increment is retrieved with the method GetInc().
            node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
            if PySpin.IsAvailable(node_offset_y) and PySpin.IsWritable(node_offset_y):

                node_offset_y.SetValue(node_offset_y.GetMin())
                print('Offset Y set to %i...' % node_offset_y.GetMin())

            else:
                print('Offset Y not available...')

            # Set maximum width
            #
            # *** NOTES ***
            # Other nodes, such as those corresponding to image width and height,
            # might have an increment other than 1. In these cases, it can be
            # important to check that the desired value is a multiple of the
            # increment. However, as these values are being set to the maximum,
            # there is no reason to check against the increment.
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):

                width_to_set = node_width.GetMax()
                node_width.SetValue(width_to_set)
                print('Width set to %i...' % node_width.GetValue())

            else:
                print('Width not available...')

            # Set maximum height
            #
            # *** NOTES ***
            # A maximum is retrieved with the method GetMax(). A node's minimum and
            # maximum should always be a multiple of its increment.
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):

                height_to_set = node_height.GetMax()
                node_height.SetValue(height_to_set)
                print('Height set to %i...' % node_height.GetValue())

            else:
                print('Height not available...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def init_cameras(self):
        result = True

        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()
        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

            print('Not enough cameras!')
            input('Done! Press Enter to exit...')
            return False

        try:
            # Retrieve transport layer nodemaps and print device information for
            # each camera
            # *** NOTES ***
            # This example retrieves information from the transport layer nodemap
            # twice: once to print device information and once to grab the device
            # serial number. Rather than caching the nodem#ap, each nodemap is
            # retrieved both times as needed.
            print('*** DEVICE INFORMATION ***\n')

            for i, cam in enumerate(self.cam_list):
                # Retrieve TL device nodemap
                nodemap_tldevice = cam.GetTLDeviceNodeMap()

                # Print device information
                # result &= self.print_device_info(nodemap_tldevice, i)

            # Initialize each camera

            # *** LATER ***
            # Each camera needs to be deinitialized once all images have been
            # acquired.
            for i, cam in enumerate(self.cam_list):
                # Initialize camera
                cam.Init()

                # Set acquisition mode to continuous
                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
                if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                    print(
                        'Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                    return False

                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                            Aborting... \n' % i)
                    return False

                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

                print('Camera %d acquisition mode set to continuous...' % i)

                # Begin acquiring images
                cam.BeginAcquisition()

                print('Camera %d started acquiring images...' % i)

                print()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        print('Done initializing!')
        print(result)

        self.started = False
        self.read_lock = threading.Lock()

    def print_device_info(self, nodemap, cam_num):
        """
        This function prints the device information of the camera from the transport
        layer; please see NodeMapInfo example for more in-depth comments on printing
        device information from the nodemap.

        :param nodemap: Transport layer device nodemap.
        :param cam_num: Camera number.
        :type nodemap: INodeMap
        :type cam_num: int
        :returns: True if successful, False otherwise.
        :rtype: bool
        """

        print('Printing device information for camera %d... \n' % cam_num)

        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                      node_feature.ToString() if PySpin.IsReadable(
                                          node_feature) else 'Node not readable'))

            else:
                print('Device control information not available.')
            print()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def get_camera_frames(self):
        with self.read_lock:
            if self.new_frames_available:
                self.new_frames_available = False
                return self.newest_frames, self.matrices
            else:
                return [], []


if __name__ == '__main__':
    DEBUG_MODE = True
    flir_blackfly_s = FlirBlackflyS()
    flir_blackfly_s.start()
