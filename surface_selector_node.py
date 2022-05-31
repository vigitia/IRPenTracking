#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import rclpy  # Python library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes

from sensor_msgs.msg import Image  # Image is the message type
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from cv_bridge import CvBridge

from .surface_selector import SurfaceSelector


class SurfaceSelectorNode(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.cv_bridge = CvBridge()

        # Set Parameters
        camera_type_parameter = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Camera type')
        #self.declare_parameter('camera_type', '/vigitia/brio_rgb_full', camera_type_parameter)

        self.declare_parameter('camera_type', '/vigitia/realsense_ir_full', camera_type_parameter)

        queue_length = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='Length of the queue')
        self.declare_parameter('queue_length', 10, queue_length)

        # Create the subscriber. This subscriber will receive an Image
        # from the /rgb/image_raw topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image,  # Datentyp
            self.get_parameter("camera_type").get_parameter_value().string_value,  # Name des Topics
            self.listener_callback,
            self.get_parameter("queue_length").get_parameter_value().integer_value)

        self.surface_selector = SurfaceSelector(self.get_parameter("camera_type").get_parameter_value().string_value)


    def listener_callback(self, data):
        """
         Callback function.
         """

        # Convert ROS Image message to OpenCV image
        current_frame = self.cv_bridge.imgmsg_to_cv2(data)

        calibration_finished = self.surface_selector.select_surface(current_frame)
        if calibration_finished:
            print("[Surface Selector Node]: Calibration Finished")
            exit()

        cv2.waitKey(1)


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    surface_selector_node = SurfaceSelectorNode()

    # Spin the node so the callback function is called.
    rclpy.spin(surface_selector_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    surface_selector_node.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()
