# Based on https://www.geeksforgeeks.org/pyqt5-create-paint-application/
import csv
import random

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys
import cv2
import datetime

import numpy as np
from scipy.spatial import distance

from realsense_d435 import RealsenseD435Camera
from ir_pen import IRPen

# from draw_shape import ShapeCreator

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

LINE_THICKNESS = 1
PEN_COLOR = QColor(255, 255, 255, 255)
LINE_COLOR = (255, 255, 255)

NUM_POINTS_IGNORE = 2  # Number of points we ignore at the beginning of a new line

realsense_d435_camera = RealsenseD435Camera()
realsense_d435_camera.init_video_capture()
realsense_d435_camera.start()

ir_pen_1 = IRPen('left camera')
ir_pen_2 = IRPen('right camera')

# To test the continuity of lines, enable this flag to cycle through different colors every time a new pen event ID
# is detected
COLOR_CYCLE_TESTING = False


PHRASES_MODE = False

PARTICIPANT_ID = 7


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

    # For testing line continuity, we assign a new color for every new ID of a pen event
    current_id = 0
    colors = [QColor(255, 255, 255, 255), QColor(0, 0, 255, 255), QColor(0, 255, 0, 255), QColor(255, 0, 0, 255),
              QColor(0, 255, 255, 255), QColor(255, 0, 255, 255), QColor(255, 255, 0, 255)]
    color_index = 0
    current_color = QColor(255, 255, 255, 255)

    def get_next_color(self):
        if self.color_index == len(self.colors):
            self.color_index = 0

        self.current_color = self.colors[self.color_index]
        self.color_index += 1

    def __init__(self, painting_widget):
        QThread.__init__(self)

        self.painting_widget = painting_widget
        # self.ir_pen = IRPen()

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            self.process_frame()
            # self.process_frame_stereo()

    def find_pen_position_subpixel(self, thresh):
        thresh_large = cv2.resize(thresh, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
        contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[0] if len(contours) == 2 else contours[1]

        if len(contours) == 0:
            return False, None

        min_radius = thresh.shape[0]
        smallest_contour = contours[0]

        # print(len(contours))
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                min_radius = radius
                smallest_contour = contour

        M = cv2.moments(smallest_contour)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])

        cY = int(M["m01"] / M["m00"])

        position = (cX, cY)

        return True, position

    def process_frame_stereo(self):
        global realsense_d435_camera
        global ir_pen_1
        global ir_pen_2

        ir_image_table, current_ir_image_table_1, current_ir_image_table_2 = realsense_d435_camera.get_ir_image()

        if ir_image_table is not None:
            active_pen_events, _, _, pen_events_to_remove = ir_pen_1.get_ir_pen_events(ir_image_table)

            if len(pen_events_to_remove) > 0:
                self.painting_widget.reset_last_point()

            new_points = []
            for active_pen_event in active_pen_events:

                if COLOR_CYCLE_TESTING:
                    if active_pen_event.id > self.current_id:
                        self.get_next_color()
                        self.current_id = active_pen_event.id

                if active_pen_event.state.value != 3:  # All events except hover
                    if len(active_pen_event.history) > NUM_POINTS_IGNORE:
                            new_points.append(QPoint(active_pen_event.x, active_pen_event.y))

            # for active_pen_event in active_pen_events_2:
            #     if active_pen_event.state.value != 3:  # All events except hover
            #         if len(active_pen_event.history) > NUM_POINTS_IGNORE:
            #             new_points.append(QPoint(active_pen_event.x, active_pen_event.y))

            if COLOR_CYCLE_TESTING:
                self.painting_widget.draw_new_points(new_points, self.current_color)
            else:
                self.painting_widget.draw_new_points(new_points)

    # @timeit("Process Frame")
    def process_frame(self):
        global realsense_d435_camera
        global ir_pen_1
        global ir_pen_2
        ir_image_table, current_ir_image_table_1, current_ir_image_table_2 = realsense_d435_camera.get_ir_image()

        if ir_image_table is not None and current_ir_image_table_1 is not None and current_ir_image_table_2 is not None:

            _, brightest_1, _, (max_x_1, max_y_1) = cv2.minMaxLoc(current_ir_image_table_1)
            _, brightest_2, _, (max_x_2, max_y_2) = cv2.minMaxLoc(current_ir_image_table_2)

            # LATENZ MESSUNG
            # if brightest_1 > 100:
            #     self.painting_widget.fill_screen(True)
            #     return
            # else:
            #     self.painting_widget.fill_screen(False)
            #     return

            MIN_BRIGHTNESS = 30

            pos = None

            active_pen_events_1, _, _, pen_events_to_remove_1 = ir_pen_1.get_ir_pen_events(
                current_ir_image_table_1)
            active_pen_events_2, _, _, pen_events_to_remove_2 = ir_pen_2.get_ir_pen_events(
                current_ir_image_table_2)

            active_pen_events = []
            pen_events_to_remove = pen_events_to_remove_1 + pen_events_to_remove_2

            _, thresh_1 = cv2.threshold(current_ir_image_table_1, int(np.median(current_ir_image_table_1) * 1.2), 255, cv2.THRESH_BINARY)
            _, thresh_2 = cv2.threshold(current_ir_image_table_2, int(np.median(current_ir_image_table_2) * 1.2), 255, cv2.THRESH_BINARY)

            dest_and = cv2.bitwise_and(thresh_1, thresh_2, mask=None)

            self.painting_widget.preview_images_on_canvas([current_ir_image_table_1, current_ir_image_table_2, ir_image_table, dest_and],
                                                          ['left ir', 'right ir', 'Fake color', 'AND'])

            max_and = np.max(dest_and)
            max_thresh_1 = np.max(thresh_1)
            max_thresh_2 = np.max(thresh_2)

            # print(brightest_1, brightest_2, np.max(dest_and), np.max(thresh_1), np.max(thresh_2))

            # if (brightest_1 < MIN_BRIGHTNESS and brightest_2 < MIN_BRIGHTNESS) or (np.max(thresh_1) == 0 and np.max(thresh_2) == 0):
            #
            # # elif brightest_1 >= MIN_BRIGHTNESS and brightest_2 < MIN_BRIGHTNESS:

            if max_and != 0:
                non_zero_px = cv2.countNonZero(dest_and)

                text = '-- both ' + str(non_zero_px)
                success, new_pos = self.find_pen_position_subpixel(dest_and)
                if success:
                    pos = new_pos

                if brightest_1 > brightest_2:
                    active_pen_events = active_pen_events_1

                else:
                    active_pen_events = active_pen_events_2
            elif (max_thresh_1 == 0 and max_thresh_2 == 0):
                text = '-- No Pen'

            elif (max_thresh_1 != 0 and max_thresh_2 != 0):
                text = 'hover far'

            elif brightest_1 >= MIN_BRIGHTNESS and max_thresh_2 == 0:
                text = '-- cam right'
                active_pen_events = active_pen_events_1

            # elif brightest_2 >= MIN_BRIGHTNESS and brightest_1 < MIN_BRIGHTNESS:
            elif brightest_2 >= MIN_BRIGHTNESS and max_thresh_1 == 0:
                text = '-- cam left'
                active_pen_events = active_pen_events_2
            else:
                text = 'We missed someting'

            # print(text)



            #
            # _, thresh_1 = cv2.threshold(current_ir_image_table_1, np.max(current_ir_image_table_1) - 120, 255, cv2.THRESH_BINARY)
            # _, thresh_2 = cv2.threshold(current_ir_image_table_2, np.max(current_ir_image_table_2) - 120, 255, cv2.THRESH_BINARY)
            #
            # dest_and = cv2.bitwise_and(thresh_1, thresh_2, mask=None)
            #
            # if np.max(dest_and) == 255:
            #     print('----------> Maybe')
            #
            #
            #
            # active_pen_events_1, stored_lines, new_lines, pen_events_to_remove_1 = ir_pen_1.get_ir_pen_events(current_ir_image_table_1)
            # active_pen_events_2, stored_lines, new_lines, pen_events_to_remove_2 = ir_pen_2.get_ir_pen_events(current_ir_image_table_2)
            #
            # dist = 0
            #
            # if len(active_pen_events_1) > 0 and len(active_pen_events_2) > 0:
            #     dist = distance.euclidean((active_pen_events_1[0].x, active_pen_events_1[0].y), (active_pen_events_2[0].x, active_pen_events_2[0].y))
            #
            #     # print('Dist:', dist)
            #
            # if brightest_1 > brightest_2:
            #     # max_x = max_x_1
            #     active_pen_events = active_pen_events_1
            # else:
            #     # max_x = max_x_2
            #     active_pen_events = active_pen_events_2


            # if max_x > 424:
            #     print('Camera 1')
            #     active_pen_events, stored_lines, new_lines, pen_events_to_remove = ir_pen.get_ir_pen_events(
            #         current_ir_image_table_1)
            # else:
            #     print('Camera 2')
            #     active_pen_events, stored_lines, new_lines, pen_events_to_remove = ir_pen.get_ir_pen_events(
            #         current_ir_image_table_2)

            # active_pen_events, stored_lines, new_lines, pen_events_to_remove = ir_pen.get_ir_pen_events(ir_image_table)
            # active_pen_events, stored_lines, new_lines, pen_events_to_remove = ir_pen.get_ir_pen_events(current_ir_image_table_1)
            # active_pen_events_2, stored_lines_2, new_lines_2, pen_events_to_remove_2 = ir_pen.get_ir_pen_events(current_ir_image_table_2)

            if len(pen_events_to_remove) > 0:
                self.painting_widget.reset_last_point()


            new_points = []
            # if 40 > dist > 0:
            for active_pen_event in active_pen_events:

                if COLOR_CYCLE_TESTING:
                    if active_pen_event.id > self.current_id:
                        self.get_next_color()
                        self.current_id = active_pen_event.id
                #
                # print(active_pen_event)
                if active_pen_event.state.value != 3:  # All events except hover
                    if len(active_pen_event.history) > NUM_POINTS_IGNORE:
                        if pos is not None:
                            new_points.append(QPoint(pos[0], pos[1]))
                        else:
                            new_points.append(QPoint(active_pen_event.x, active_pen_event.y))



            # for active_pen_event in active_pen_events_2:
            #     if active_pen_event.state.value != 3:  # All events except hover
            #         if len(active_pen_event.history) > NUM_POINTS_IGNORE:
            #             new_points.append(QPoint(active_pen_event.x, active_pen_event.y))

            if COLOR_CYCLE_TESTING:
                self.painting_widget.draw_new_points(new_points, text, self.current_color)
            else:
                self.painting_widget.draw_new_points(new_points, text)


class PaintingWidget(QMainWindow):

    # TODO: Change this to work with multiple pens at once
    lastPoint = None

    mackenzie_phrases = []

    height_multiplier = 2

    num_phrases_written = 0

    preview_image = None

    def __init__(self):
        super().__init__()

        # shape_creator = ShapeCreator(WINDOW_WIDTH, WINDOW_HEIGHT)
        # background_image = shape_creator.draw_shape('shapes/wave.svg', (800, 800), 1000, LINE_THICKNESS, ShapeCreator.DASH, LINE_COLOR)
        # self.background_image = QImage(background_image, background_image.shape[1], background_image.shape[0], background_image.shape[1] * 3, QImage.Format_RGB888)

        self.read_mackenzie_phrases()

        # drawing flag
        self.drawing = False

        self.line_thickness = LINE_THICKNESS

        self.initUI()

        self.thread = ApplicationLoopThread(self)
        self.thread.start()



    def initUI(self):
        self.showFullScreen()  # Application should run in Fullscreen

        # setting geometry to main window
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Define width and height as global variables
        # self.width = QApplication.desktop().screenGeometry().width()
        # self.height = QApplication.desktop().screenGeometry().height()

        # hover_cursor = QLabel(self)
        # pixmap = QPixmap("testing.png")
        # # pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        # hover_cursor.setPixmap(pixmap)
        # hover_cursor.move(100, 100)
        #
        # label = QLabel(text="Welcome to Python GUI!")
        # label.show()

        self.reset_image()

    def read_mackenzie_phrases(self):
        with open('phrases.txt') as file:
            lines = file.readlines()
            for line in lines:
                self.mackenzie_phrases.append(line.replace('\n', ''))

    def get_random_phrase(self):

        index = random.randint(0, len(self.mackenzie_phrases) - 1)

        random_phrase = self.mackenzie_phrases.pop(index)

        return random_phrase


    def reset_image(self):
        # self.image = self.background_image.copy()
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        self.image.fill(Qt.black)

        if PHRASES_MODE:
            self.draw_rectangle()

    def fill_screen(self, fill):
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        if fill:
            self.image.fill(Qt.white)
        else:
            self.image.fill(Qt.black)
        self.update()

    def draw_new_points(self, points, text, color=PEN_COLOR):

        if 'left' in text:
            color = QColor(255, 0, 255, 255)
        if 'right' in text:
            color = QColor(255, 255, 0, 255)

        # print('drawing', points)

        painter = QPainter(self.image)
        # painter.setRenderHint(QPainter.Antialiasing)

        RECTANGLE_COLOR = QColor(40, 40, 40, 255)

        # painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(RECTANGLE_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(RECTANGLE_COLOR))
        painter.drawRect(80, 500, 600, 100)

        painter.setPen(QPen(color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        font = QFont()
        font.setFamily('Arial')
        # font.setBold(True)
        font.setPointSize(40)
        painter.setFont(font)

        painter.drawText(100, 600, text)

        for point in points:
            if self.lastPoint is None:
                painter.drawPoint(point)
            else:
                painter.drawLine(self.lastPoint, point)
            self.lastPoint = point

        self.update()

    def draw_rectangle(self):
        painter = QPainter(self.image)

        painter.setPen(QPen(Qt.white))

        font = QFont()
        font.setFamily('Arial')
        # font.setBold(True)
        font.setPointSize(80)
        painter.setFont(font)

        painter.drawText(100, 300, self.get_random_phrase())

        RECTANGLE_COLOR = QColor(50, 50, 50, 255)

        # painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(RECTANGLE_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(RECTANGLE_COLOR))

        if self.height_multiplier > 3:
            self.height_multiplier = 2

        height = 58 * self.height_multiplier
        width = WINDOW_WIDTH/2

        self.height_multiplier += 1

        painter.drawRect(int(WINDOW_WIDTH/2 - width/2), int(WINDOW_HEIGHT/2 - height/2), width, height)

        self.update()

    def reset_last_point(self):
        print('##########################################')
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
            painter.setPen(QPen(PEN_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

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
            self.save_screenshot()
            self.close()
            sys.exit()
        elif event.key() == Qt.Key_Space:
            self.save_screenshot()

            if PHRASES_MODE:
                self.num_phrases_written += 1
                if self.num_phrases_written == 10:
                    print('10 Phrases have been written')
                    self.close()
                    sys.exit()

            self.reset_image()

    # Save the current window content as a png
    def save_screenshot(self):
        filename = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
        self.image.save('./output_images/Participant_{}_{}.png'.format(PARTICIPANT_ID, filename))

    def preview_images_on_canvas(self, images_to_preview, image_names):
        painter = QPainter(self.image)

        x_anchor = 3840

        for i, image_to_preview in enumerate(images_to_preview):
            if len(image_to_preview.shape) < 3:
                image_to_preview = cv2.cvtColor(image_to_preview, cv2.COLOR_GRAY2RGB)
            image_to_preview = cv2.putText(
                img=image_to_preview,
                text=image_names[i],
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(255, 255, 255),
                thickness=2
            )
            image_to_preview = cv2.copyMakeBorder(image_to_preview, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,255,255))
            preview_image = QImage(image_to_preview.data, image_to_preview.shape[1], image_to_preview.shape[0],
                                   QImage.Format_RGB888)

            x_anchor -= image_to_preview.shape[1]

            painter.drawImage(x_anchor, 0, preview_image)

        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = PaintingWidget()
    window.show()
    sys.exit(app.exec_())
