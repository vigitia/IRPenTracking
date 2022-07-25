#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from sensor_msgs.msg import Image

from message_filters import TimeSynchronizer, Subscriber

from rclpy.executors import MultiThreadedExecutor


from cv_bridge import CvBridge

from .ir_pen import IRPen


class IRPenNode(Node):

    def __init__(self):
        super().__init__('ir_pen_node')

        self.init_subscription()
        self.init_publisher()

        self.ir_pen = IRPen()

    def init_subscription(self):
        topic_parameter_flir_blackfly_s_0 = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='TODO')
        self.declare_parameter('topic_parameter_flir_blackfly_s_0', '/vigitia/flir_blackfly_s_0', topic_parameter_flir_blackfly_s_0)

        topic_parameter_flir_blackfly_s_1 = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='TODO')
        self.declare_parameter('topic_parameter_flir_blackfly_s_1', '/vigitia/flir_blackfly_s_1',
                               topic_parameter_flir_blackfly_s_1)

        queue_length = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='Length of the queue')
        self.declare_parameter('queue_length', 10, queue_length)

        # self.subscription = self.create_subscription(
        #     Image,  # Datentyp
        #     self.get_parameter("topic_parameter_flir_blackfly_s").get_parameter_value().string_value,
        #     self.listener_callback,
        #     self.get_parameter("queue_length").get_parameter_value().integer_value)

        # Create the Subscribers
        self.subscription_flir_blackfly_s_0 = Subscriber(
            self,
            Image,  # Datentyp
            self.get_parameter("topic_parameter_flir_blackfly_s_0").get_parameter_value().string_value,  # Name des Topics
            qos_profile=self.get_parameter("queue_length").get_parameter_value().integer_value)

        self.subscription_flir_blackfly_s_1 = Subscriber(
            self,
            Image,  # Datentyp
            self.get_parameter("topic_parameter_flir_blackfly_s_1").get_parameter_value().string_value,  # Name des Topics
            qos_profile=self.get_parameter("queue_length").get_parameter_value().integer_value)

        self.synchronizer = TimeSynchronizer([self.subscription_flir_blackfly_s_0, self.subscription_flir_blackfly_s_1],
                                             self.get_parameter("queue_length").get_parameter_value().integer_value)
        self.synchronizer.registerCallback(self.sychronized_callback)


    def init_publisher(self):
        param_desc_topic = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                               description='name of publishing topic')
        self.declare_parameter('topic', '/vigitia/ir_pen_events', param_desc_topic)

        self.cv_bridge = CvBridge()

        self.publisher = self.create_publisher(msg_type=Image,
                                               topic=self.get_parameter("topic").get_parameter_value().string_value,
                                               qos_profile=self.get_parameter("queue_length").get_parameter_value().
                                               integer_value)

    def sychronized_callback(self, flir_blackfly_s_0, flir_blackfly_s_1):
        frame_0 = self.cv_bridge.imgmsg_to_cv2(flir_blackfly_s_0)
        frame_1 = self.cv_bridge.imgmsg_to_cv2(flir_blackfly_s_1)

        print(frame_0.shape, frame_1.shape)

        # active_pen_events, stored_lines, _, _, debug_distances = self.ir_pen.get_ir_pen_events_multicam([frame_0, frame_1], matrices)



def main(args=None):
    rclpy.init(args=args)
    ir_pen_node = IRPenNode()
    # executor = MultiThreadedExecutor()
    # ir_pen_node.get_logger().info("Shutting down")
    # ir_pen_node.destroy_node()
    # executor.shutdown()

    rclpy.spin(ir_pen_node)
    ir_pen_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
