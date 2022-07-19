#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .flir_blackfly_s import FlirBlackflyS

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from sensor_msgs.msg import Image


from cv_bridge import CvBridge


class FLirBlackflySNode(Node):

    def __init__(self):
        super().__init__('flir_blackfly_s_node')

        param_desc_topic = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                               description='name of publishing topic')
        self.declare_parameter('topic', '/vigitia/flir_blackfly_s', param_desc_topic)

        param_desc_queue_length = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                      description='length of the queue')
        self.declare_parameter('queue_length', 10, param_desc_queue_length)

        self.cv_bridge = CvBridge()

        self.publisher = self.create_publisher(msg_type=Image,
                                               topic=self.get_parameter("topic").get_parameter_value().string_value,
                                               qos_profile=self.get_parameter("queue_length").get_parameter_value().
                                               integer_value)

        flir_blackfly_s = FlirBlackflyS(ros2_node=self)
        flir_blackfly_s.start()




def main(args=None):
    rclpy.init(args=args)
    flir_blackfly_s_node = FLirBlackflySNode()
    rclpy.spin(flir_blackfly_s_node)
    flir_blackfly_s_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
