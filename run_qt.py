# Based on https://www.geeksforgeeks.org/pyqt5-create-paint-application/


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
import cv2
import datetime

import numpy as np

from realsense_d435 import RealsenseD435Camera
from ir_pen import IRPen

from draw_shape import ShapeCreator

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

LINE_THICKNESS = 1
PEN_COLOR = QColor(255, 255, 255, 255)
LINE_COLOR = (255, 255, 255)

NUM_POINTS_IGNORE = 2  # Number of points we ignore at the beginning of a new line

realsense_d435_camera = RealsenseD435Camera()
realsense_d435_camera.init_video_capture()
realsense_d435_camera.start()

ir_pen = IRPen()


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            # print("I " + prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            # print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) + " ms)")
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return retval
        return wrapper
    return timeit_decorator


class ApplicationLoopThread(QThread):

    def __init__(self, painting_widget):
        QThread.__init__(self)

        self.painting_widget = painting_widget
        # self.ir_pen = IRPen()

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            self.process_frame()

    # @timeit("Process Frame")
    def process_frame(self):
        global realsense_d435_camera
        global ir_pen
        ir_image_table = realsense_d435_camera.get_ir_image()

        if ir_image_table is not None:

            active_pen_events, stored_lines, new_lines, pen_events_to_remove = ir_pen.get_ir_pen_events(ir_image_table)

            if len(pen_events_to_remove) > 0:
                self.painting_widget.reset_last_point()

            new_points = []
            for active_pen_event in active_pen_events:
                # print(active_pen_event)
                if active_pen_event.state.value != 3:  # All events except hover
                    if len(active_pen_event.history) > NUM_POINTS_IGNORE:
                        new_points.append(QPoint(active_pen_event.x, active_pen_event.y))

            self.painting_widget.draw_new_points(new_points)

        #     new_frame = self.base_image.copy()
        #
        #     for active_pen_event in active_pen_events:
        #         print(active_pen_event)
        #         if active_pen_event.state.value == 3:  # HOVER
        #             cv2.circle(new_frame, active_pen_event.get_coordinates(), 5, (0, 255, 0))
        #         else:
        #             line = np.array(active_pen_event.history, np.int32)
        #             cv2.polylines(new_frame, [line], isClosed=False, color=PEN_COLOR, thickness=LINE_THICKNESS)
        #
        #     for line in stored_lines:
        #         line = np.array(line, np.int32)
        #         cv2.polylines(new_frame, [line], isClosed=False, color=PEN_COLOR, thickness=LINE_THICKNESS)
        #
        #     # print(ir_image_table.shape)
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     sys.exit(0)


class PaintingWidget(QMainWindow):

    # TODO: Change this to work with multiple pens at once
    lastPoint = None

    def __init__(self):
        super().__init__()

        shape_creator = ShapeCreator(WINDOW_WIDTH, WINDOW_HEIGHT)
        background_image = shape_creator.draw_shape('shapes/wave.svg', (800, 800), 1000, LINE_THICKNESS, ShapeCreator.DASH, LINE_COLOR)
        self.background_image = QImage(background_image, background_image.shape[1], background_image.shape[0], background_image.shape[1] * 3, QImage.Format_RGB888)

        self.initUI()

        # drawing flag
        self.drawing = False

        self.line_thickness = LINE_THICKNESS
        self.pen_color = PEN_COLOR

        self.thread = ApplicationLoopThread(self)
        self.thread.start()

    def initUI(self):
        self.showFullScreen()  # Application should run in Fullscreen

        # Define width and height as global variables
        # self.width = QApplication.desktop().screenGeometry().width()
        # self.height = QApplication.desktop().screenGeometry().height()

        # setting geometry to main window
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.reset_image()

    def reset_image(self):
        self.image = self.background_image
        #self.image = QImage(self.size(), QImage.Format_ARGB32)
        #self.image.fill(Qt.white)

    def draw_new_points(self, points):

        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        for point in points:
            if self.lastPoint is None:
                painter.drawPoint(point)
            else:
                painter.drawLine(self.lastPoint, point)
            self.lastPoint = point

        self.update()

    def reset_last_point(self):
        self.lastPoint = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(QPen(self.pen_color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            # painter.drawLine(self.lastPoint, event.pos())
            painter.drawPoint(event.pos())

            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    # Handle Key-press events
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = PaintingWidget()
    window.show()
    sys.exit(app.exec_())
