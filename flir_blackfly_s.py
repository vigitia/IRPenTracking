import sys

import PySpin
import cv2

global continue_recording
continue_recording = True


class FlirBlackflyS:

    def __init__(self):
        self.init_cameras()

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
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            cam_list.Clear()

            # Release system instance
            system.ReleaseInstance()

            print('Not enough cameras!')
            input('Done! Press Enter to exit...')
            return False

        result = self.run_cameras(cam_list)

        # # Run example on each camera
        # for i, cam in enumerate(cam_list):
        #     print('Running example for camera %d...' % i)
        #
        #     result &= self.run_single_camera(cam)
        #     print('Camera %d example complete... \n' % i)
        #
        # # Release reference to camera
        # # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # # cleaned up when going out of scope.
        # # The usage of del is preferred to assigning the variable to None.
        # del cam

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        input('Done! Press Enter to exit...')
        return result

    def run_cameras(self, cam_list):
        """
            This function acts as the body of the example; please see NodeMapInfo example
            for more in-depth comments on setting up cameras.

            :param cam_list: List of cameras
            :type cam_list: CameraList
            :return: True if successful, False otherwise.
            :rtype: bool
            """
        try:
            result = True

            # Retrieve transport layer nodemaps and print device information for
            # each camera
            # *** NOTES ***
            # This example retrieves information from the transport layer nodemap
            # twice: once to print device information and once to grab the device
            # serial number. Rather than caching the nodem#ap, each nodemap is
            # retrieved both times as needed.
            print('*** DEVICE INFORMATION ***\n')

            for i, cam in enumerate(cam_list):
                # Retrieve TL device nodemap
                nodemap_tldevice = cam.GetTLDeviceNodeMap()

                # Print device information
                result &= self.print_device_info(nodemap_tldevice, i)

            # Initialize each camera
            #
            # *** NOTES ***
            # You may notice that the steps in this function have more loops with
            # less steps per loop; this contrasts the AcquireImages() function
            # which has less loops but more steps per loop. This is done for
            # demonstrative purposes as both work equally well.
            #
            # *** LATER ***
            # Each camera needs to be deinitialized once all images have been
            # acquired.
            for i, cam in enumerate(cam_list):
                # Initialize camera
                cam.Init()

            # Acquire images on all cameras
            # result &= self.acquire_images(cam_list)
            result &= self.acquire_images_multiple(cam_list)

            # Deinitialize each camera
            #
            # *** NOTES ***
            # Again, each camera must be deinitialized separately by first
            # selecting the camera and then deinitializing it.
            for cam in cam_list:
                # Deinitialize camera
                cam.DeInit()

            # Release reference to camera
            # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
            # cleaned up when going out of scope.
            # The usage of del is preferred to assigning the variable to None.
            del cam

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

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

    # def run_single_camera(self, cam):
    #     """
    #     This function acts as the body of the example; please see NodeMapInfo example
    #     for more in-depth comments on setting up cameras.
    #
    #     :param cam: Camera to run on.
    #     :type cam: CameraPtr
    #     :return: True if successful, False otherwise.
    #     :rtype: bool
    #     """
    #     try:
    #         result = True
    #
    #         nodemap_tldevice = cam.GetTLDeviceNodeMap()
    #
    #         # Initialize camera
    #         cam.Init()
    #
    #         # Retrieve GenICam nodemap
    #         nodemap = cam.GetNodeMap()
    #
    #         # if not self.configure_custom_image_settings(cam):
    #         #     return False
    #
    #         # Acquire images
    #         result &= self.acquire_images(cam, nodemap, nodemap_tldevice)
    #
    #         # Deinitialize camera
    #         cam.DeInit()
    #
    #     except PySpin.SpinnakerException as ex:
    #         print('Error: %s' % ex)
    #         result = False
    #
    #     return result

    def acquire_images_multiple(self, cam_list):
        global continue_recording

        try:
            result = True

            # Prepare each camera to acquire images
            #
            # *** NOTES ***
            # For pseudo-simultaneous streaming, each camera is prepared as if it
            # were just one, but in a loop. Notice that cameras are selected with
            # an index. We demonstrate pseduo-simultaneous streaming because true
            # simultaneous streaming would require multiple process or threads,
            # which is too complex for an example.
            #

            for i, cam in enumerate(cam_list):

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

            while continue_recording:
                for i, cam in enumerate(cam_list):
                    try:
                        # Retrieve next received image and ensure image completion
                        image_result = cam.GetNextImage(1000)

                        #  Ensure image completion
                        if image_result.IsIncomplete():
                            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                        else:
                            # Getting the image data as a numpy array
                            image_data = image_result.GetNDArray()

                            width = image_result.GetWidth()
                            height = image_result.GetHeight()
                            print('Camera %d grabbed image, width = %d, height = %d' % (i, width, height))

                            # Convert image to mono 8
                            # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                            # Save image
                            # image_converted.Save(filename)
                            # print('Image saved at %s' % filename)

                            # print(image_data.shape)
                            cv2.imshow('Flir Camera {}'.format(i), image_data)
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC
                                continue_recording = False
                                cv2.destroyAllWindows()
                                sys.exit(0)

                        # Release image
                        image_result.Release()

                    except PySpin.SpinnakerException as ex:
                        print('Error: %s' % ex)
                        result = False

            # # Retrieve, convert, and save images for each camera
            # #
            # # *** NOTES ***
            # # In order to work with simultaneous camera streams, nested loops are
            # # needed. It is important that the inner loop be the one iterating
            # # through the cameras; otherwise, all images will be grabbed from a
            # # single camera before grabbing any images from another.
            # for i, cam in enumerate(cam_list):
            #     try:
            #         # Retrieve device serial number for filename
            #         node_device_serial_number = PySpin.CStringPtr(
            #             cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
            #
            #         if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(
            #                 node_device_serial_number):
            #             device_serial_number = node_device_serial_number.GetValue()
            #             print('Camera %d serial number set to %s...' % (i, device_serial_number))
            #
            #         # Retrieve next received image and ensure image completion
            #         image_result = cam.GetNextImage(1000)
            #
            #         #  Ensure image completion
            #         if image_result.IsIncomplete():
            #             print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            #
            #         else:
            #             # Getting the image data as a numpy array
            #             image_data = image_result.GetNDArray()
            #
            #             width = image_result.GetWidth()
            #             height = image_result.GetHeight()
            #             print('Camera %d grabbed image, width = %d, height = %d' % (i, width, height))
            #
            #             # Convert image to mono 8
            #             # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            #
            #             # Save image
            #             # image_converted.Save(filename)
            #             # print('Image saved at %s' % filename)
            #
            #             # print(image_data.shape)
            #             cv2.imshow('Flir Camera {}'.format(i), image_data)
            #             key = cv2.waitKey(1)
            #             if key == 27:  # ESC
            #                 continue_recording = False
            #                 cv2.destroyAllWindows()
            #                 sys.exit(0)
            #
            #         # Release image
            #         image_result.Release()
            #         print()
            #
            #     except PySpin.SpinnakerException as ex:
            #         print('Error: %s' % ex)
            #         result = False

            # End acquisition for each camera
            #
            # *** NOTES ***
            # Notice that what is usually a one-step process is now two steps
            # because of the additional step of selecting the camera. It is worth
            # repeating that camera selection needs to be done once per loop.
            #
            # It is possible to interact with cameras through the camera list with
            # GetByIndex(); this is an alternative to retrieving cameras as
            # CameraPtr objects that can be quick and easy for small tasks.
            for cam in cam_list:
                # End acquisition
                cam.EndAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def acquire_images(self, cam, nodemap, nodemap_tldevice):
        global continue_recording

        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()

            print('Acquiring images')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s' % device_serial_number)


            # Retrieve and display images
            while continue_recording:
                try:

                    #  Retrieve next received image
                    #
                    #  *** NOTES ***
                    #  Capturing an image houses images on the camera buffer. Trying
                    #  to capture an image that does not exist will hang the camera.
                    #
                    #  *** LATER ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.

                    image_result = cam.GetNextImage(1000)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:
                        # Getting the image data as a numpy array
                        image_data = image_result.GetNDArray()

                        # print(image_data.shape)
                        cv2.imshow('Flir', image_data)
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC
                            continue_recording = False
                            cv2.destroyAllWindows()
                            sys.exit(0)

                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
            cam.EndAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True


if __name__ == '__main__':
    flir_blackfly_s = FlirBlackflyS()
    sys.exit(0)
