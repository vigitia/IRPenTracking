
import sys
import datetime

import cv2
from PyQt5.QtCore import Qt, QPoint, QThread
from PyQt5.QtGui import (QBrush, QColor, QPainter, QPen, QSurfaceFormat, QPolygon, QFont, QImage)
from PyQt5.QtWidgets import (QApplication, QOpenGLWidget, QMainWindow)

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor

from vigitia_interfaces.msg import IRPenEvents

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

LINE_THICKNESS = 1


class ApplicationLoopThread(QThread):

    def __init__(self):
        QThread.__init__(self)

        rclpy.init()
        self.pen_drawing_node = PenDrawingNode()

    def __del__(self):
        self.wait()
        self.pen_drawing_node.destroy_node()
        rclpy.shutdown()

    def run(self):
        while True:
            rclpy.spin_once(self.pen_drawing_node)


class PenDrawingNode(Node):

    def __init__(self):
        super().__init__('pen_drawing_node')

        self.init_subscription()

    def init_subscription(self):
        topic_parameter_ir_pen_events = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='TODO')
        self.declare_parameter('topic_parameter_pen_events', '/vigitia/ir_pen_events',
                               topic_parameter_ir_pen_events)

        queue_length = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='Length of the queue')
        self.declare_parameter('queue_length', 10, queue_length)

        self.subscription = self.create_subscription(
            IRPenEvents,  # Datentyp
            self.get_parameter("topic_parameter_ir_pen_events").get_parameter_value().string_value,
            self.listener_callback,
            self.get_parameter("queue_length").get_parameter_value().integer_value)

    def listener_callback(self, data):
        print(data)

class GLWidget(QOpenGLWidget):

    active_pen_events = []
    stored_lines = []

    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)

        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.background = QBrush(QColor(0, 0, 0))

        self.background_image = QImage(self.size(), QImage.Format_RGB32)
        self.background_image.fill(Qt.black)

        self.pen = QPen(Qt.white)
        self.pen.setWidth(LINE_THICKNESS)

        self.font = QFont()
        # self.font.setFamily('Arial')
        # self.font.setBold(True)
        self.font.setPixelSize(40)

        self.color = QColor(255, 255, 255, 255)
        self.color_hover = QColor(255, 255, 255, 40)

        self.setAutoFillBackground(False)

        self.last_stored_lines_length = 0

    def update_data(self, active_pen_events, stored_lines):
        self.active_pen_events = active_pen_events
        self.stored_lines = stored_lines

    # @timeit("Paint")
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        #painter.setRenderHint(QPainter.Antialiasing)

        #painter.fillRect(event.rect(), self.background)
        painter.drawImage(event.rect(), self.background_image, self.background_image.rect())

        painter.setPen(self.pen)

        # global last_time
        # global current_debug_distances
        # painter.setFont(self.font)
        # painter.drawText(current_debug_distances[1][0], current_debug_distances[1][1], str(current_debug_distances[0]))
        # painter.drawText(100, 500, str(last_time) + ' ms')

        polygons_to_draw = []

        for active_pen_event in self.active_pen_events:

            polygon = []

            if active_pen_event.state.value != 3:  # All events except hover
                for point in active_pen_event.history:
                    polygon.append(QPoint(point[0], point[1]))
            # else:
            #     # Draw a dot to show hover events
            #     painter.setPen(QPen(self.color_hover, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            #     painter.setBrush(QBrush(self.color_hover, Qt.SolidPattern))
            #     painter.drawEllipse(active_pen_event.x, active_pen_event.y, 5, 5)
            #
            #     painter.setPen(QPen(self.color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            if len(polygon) > 0:
                polygons_to_draw.append(polygon)

        if len(self.stored_lines) > self.last_stored_lines_length:
            background_painter = QPainter(self.background_image)
            background_painter.begin(self)
            background_painter.setPen(self.pen)

            #for line in self.stored_lines:
            for i in range(self.last_stored_lines_length, len(self.stored_lines)):
                line = self.stored_lines[i]

                polygon = []

                for point in line:
                    polygon.append(QPoint(point[0], point[1]))

                if len(polygon) > 0:
                    background_painter.drawPolyline(QPolygon(polygon))
            self.last_stored_lines_length = len(self.stored_lines)

            background_painter.end()

        for polygon in polygons_to_draw:
            # painter.drawPolygon(QPolygon(polygon))
            painter.drawPolyline(QPolygon(polygon))

        painter.end()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.showFullScreen()
        self.openGL = GLWidget(self)
        self.setCentralWidget(self.openGL)

        self.thread = ApplicationLoopThread(self)
        self.thread.start()

    def draw_all_points(self, active_pen_events, stored_lines):
        self.openGL.update_data(active_pen_events, stored_lines)
        self.openGL.update()

    # Handle Key-press events
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # self.save_screenshot()

            self.close()
            sys.exit(0)
        # elif event.key() == Qt.Key_Space:
        #     self.save_screenshot()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = Window()
    window.show()
    sys.exit(app.exec_())
