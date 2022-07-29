#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from .flir_blackfly_s import FlirBlackflyS

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from sensor_msgs.msg import Image
from std_msgs.msg import String


from cv_bridge import CvBridge


class FLirBlackflySNode(Node):

    def __init__(self):
        super().__init__('flir_blackfly_s_node')

        param_desc_flir_blackfly_s_0 = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                               description='name of publishing topic')
        self.declare_parameter('flir_blackfly_s_0', '/vigitia/flir_blackfly_s_0', param_desc_flir_blackfly_s_0)

        param_desc_flir_blackfly_s_1 = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                           description='name of publishing topic')
        self.declare_parameter('flir_blackfly_s_1', '/vigitia/flir_blackfly_s_1', param_desc_flir_blackfly_s_1)

        param_desc_queue_length = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                      description='length of the queue')
        self.declare_parameter('queue_length', 10, param_desc_queue_length)

        self.cv_bridge = CvBridge()

        self.publisher_flir_blackfly_s_0 = self.create_publisher(msg_type=String,
                                                                 topic=self.get_parameter(
                                                                     'flir_blackfly_s_0').get_parameter_value().string_value,
                                                                 qos_profile=self.get_parameter(
                                                                     "queue_length").get_parameter_value().
                                                                 integer_value)

        self.publisher_flir_blackfly_s_1 = self.create_publisher(msg_type=String,
                                                                 topic=self.get_parameter('flir_blackfly_s_1').get_parameter_value().string_value,
                                                                 qos_profile=self.get_parameter("queue_length").get_parameter_value().
                                                                 integer_value)

        flir_blackfly_s = FlirBlackflyS(ros2_node=self)

        self.loop()

    def loop(self):
        print('start')
        while True:
            print('Alive')
            time.sleep(1)




def main(args=None):
    rclpy.init(args=args)
    flir_blackfly_s_node = FLirBlackflySNode()
    rclpy.spin(flir_blackfly_s_node)
    flir_blackfly_s_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
