import random
import sys
import datetime
import threading
import time

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (QBrush, QColor, QPainter, QPen, QPolygon, QFont, QImage, QSurfaceFormat, QPixmap)
from PyQt5.QtWidgets import (QApplication, QOpenGLWidget, QMainWindow)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer

from flir_blackfly_s import FlirBlackflyS
from ir_pen import IRPen

import cv2

# from draw_shape import ShapeCreator

DISPLAY_MODE = "4k"

if DISPLAY_MODE == "4k":
    WINDOW_WIDTH = 3840  # 1920
    WINDOW_HEIGHT = 2160  # 1080
    REDRAW_INTERVAL = 17
elif DISPLAY_MODE == "1080":
    WINDOW_WIDTH = 1920 # 3840  # 1920
    WINDOW_HEIGHT = 1080 # 2160  # 1080
    REDRAW_INTERVAL = 5

LINE_THICKNESS = 1
PEN_COLOR = QColor(255, 255, 255, 255)
LINE_COLOR = (255, 255, 255)

PHRASES_MODE = False
PARTICIPANT_ID = 7

last_time = 0
current_debug_distances = []

mackenzie_phrases = []
current_phrase = ""
num_phrases_written = 0

def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            global last_time
            start_time = datetime.datetime.now()
            # print("I " + prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            # print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) + " ms)")
            # if run_time > 0.1:
            last_time = prefix + ' ' + f'{run_time:.3f}'
            print(f'{prefix} > {last_time}  ms', flush=True)
            return retval
        return wrapper
    return timeit_decorator


class GLWidget(QOpenGLWidget):

    active_pen_events = []
    stored_lines = []

    fill_screen_white = False

    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setSwapInterval(0)

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

        self.redraw_timer = QTimer()
        self.redraw_timer.setInterval(REDRAW_INTERVAL)
        self.redraw_timer.timeout.connect(self.repaint)
        self.redraw_timer.start()

        #self.setAutoFillBackground(False)

        #self.redraw.connect(self._redraw())

        self.last_stored_lines_length = 0

    def mousePressEvent(self, QMouseEvent):
        self.fill_screen_white = True
        #self.repaint()

    def mouseReleaseEvent(self, QMouseEvent):
        self.fill_screen_white = False
        #self.repaint()

    def update_data(self, active_pen_events, stored_lines):
        self.active_pen_events = active_pen_events
        self.stored_lines = stored_lines

    # @pyqtSlot()
    # def redraw(self):
    #     #self.update()
    #     #self.repaint()
    #     pass

    # @timeit("Paint")
    def paintEvent(self, event):
        start_time = datetime.datetime.now()

        painter = QPainter()
        painter.begin(self)

        painter.setRenderHint(QPainter.Antialiasing)

        #painter.fillRect(event.rect(), self.background)
        painter.drawImage(event.rect(), self.background_image, self.background_image.rect())

        painter.setPen(self.pen)

        # global last_time
        # global current_debug_distances
        # painter.setFont(self.font)
        # painter.drawText(current_debug_distances[1][0], current_debug_distances[1][1], str(current_debug_distances[0]))
        # painter.drawText(100, 500, str(last_time) + ' ms')

        if self.fill_screen_white:
            painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
            painter.drawRect(600, 0, 1900, 1000)
            painter.end()
            return

        # global current_debug_distances
        # # for point in current_debug_distances:
        #     # print(point)
        # if len(current_debug_distances) > 0:
        #     painter.drawEllipse(QPoint(int(current_debug_distances[0][0]), int(current_debug_distances[0][1])), 5, 5)


        if PHRASES_MODE:
            painter.setFont(self.font)
            global current_phrase
            painter.drawText(WINDOW_WIDTH / 2 - ((len(current_phrase) * 20) / 2), 300, current_phrase)

            RECTANGLE_COLOR = QColor(50, 50, 50, 255)

            # painter.setPen(QPen(RECTANGLE_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QBrush(RECTANGLE_COLOR))
            height = 120
            width = WINDOW_WIDTH/2
            painter.drawRect(int(WINDOW_WIDTH/2 - width/2), int(WINDOW_HEIGHT/2 - height/2), width, height)


        polygons_to_draw = []

        for active_pen_event in self.active_pen_events:

            polygon = []

            # TODO: Maybe remove this if because the points in history might still be relevant?
            # if active_pen_event.state.value != 3:  # All events except hover
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

        # TODO: Empty the stored lines list after the line is drawn
        if len(self.stored_lines) > self.last_stored_lines_length:
            print('Store LINE')

            for i in range(self.last_stored_lines_length, len(self.stored_lines)):
                line = self.stored_lines[i]

                draw_background_thread = threading.Thread(target=self.draw_background, args=(line,), daemon=True)
                draw_background_thread.start()

                polygon = []

                for point in line:
                    polygon.append(QPoint(point[0], point[1]))

                if len(polygon) > 0:
                    polygons_to_draw.append(polygon)

        for polygon in polygons_to_draw:
            painter.drawPolyline(QPolygon(polygon))

        painter.end()

        end_time = datetime.datetime.now()
        run_time = (end_time - start_time).microseconds / 1000.0

        print(run_time)

    def draw_background(self, line):
        background_painter = QPainter(self.background_image)
        background_painter.begin(self)
        background_painter.setPen(self.pen)

        polygon = []
        for point in line:
            polygon.append(QPoint(point[0], point[1]))

        if len(polygon) > 0:
            background_painter.drawPolyline(QPolygon(polygon))

        self.last_stored_lines_length += 1

        background_painter.end()


class Window(QMainWindow):

    start_time = time.time()
    frame_counter = 0

    rois = []

    #redraw = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.read_mackenzie_phrases()

        self.showFullScreen()
        self.openGL = GLWidget(self)
        self.setCentralWidget(self.openGL)

        #self.redraw.connect(self.openGL.redraw)

        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

    def read_mackenzie_phrases(self):
        global mackenzie_phrases
        global current_phrase
        with open('phrase_set/phrases.txt') as file:
            lines = file.readlines()
            for line in lines:
                mackenzie_phrases.append(line.replace('\n', ''))

        current_phrase = self.get_random_phrase()

    def get_random_phrase(self):
        global mackenzie_phrases
        index = random.randint(0, len(mackenzie_phrases) - 1)
        random_phrase = mackenzie_phrases.pop(index)
        return random_phrase

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):

        if len(frames) > 0:
            # self.frame_counter += 1
            # # print('Received {} new frames from Flir Blackfly S'.format(len(frames)))
            # print('run', frames[0].shape)
            #
            # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(frames[0])
            # if brightest > 100:
            #     self.fill_screen_white()
            # else:
            #     self.fill_screen_black()

            active_pen_events, stored_lines, _, _, debug_distances, rois = self.ir_pen.get_ir_pen_events_multicam(frames, matrices)

            global current_debug_distances
            current_debug_distances = debug_distances

            # self.draw_all_points(active_pen_events, stored_lines)  # 1.2 ms
            self.openGL.update_data(active_pen_events, stored_lines)
        else:
            print('No frames')

        # if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
        #     print("FPS in run_qt.py(): %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
        #     self.frame_counter = 0
        #     self.start_time = time.time()

    # def draw_all_points(self, active_pen_events, stored_lines):
    #     self.openGL.update_data(active_pen_events, stored_lines)
    #     #self.redraw.emit()

    def fill_screen_white(self):
        # For latency measurements
        self.openGL.fill_screen_white = True
        #self.redraw.emit()

    def fill_screen_black(self):
        # For latency measurements
        self.openGL.fill_screen_white = False
        #self.redraw.emit()

    # Save the current window content as a png
    def save_screenshot(self):
        filename = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
        self.openGL.background_image.save('./study_results/Participant_{}_{}.png'.format(PARTICIPANT_ID, filename))

    # Handle Key-press events
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.save_screenshot()

            self.close()
            self.flir_blackfly_s.end_camera_capture()
            sys.exit(0)
        elif event.key() == Qt.Key_Space:
            self.save_screenshot()
            global current_phrase
            current_phrase = self.get_random_phrase()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    # fmt = QSurfaceFormat()
    # fmt.setSamples(4)
    # QSurfaceFormat.setDefaultFormat(fmt)

    window = Window()
    window.show()
    sys.exit(app.exec_())






# class ApplicationLoopThread(QThread):
#
#     # For testing line continuity, we assign a new color for every new ID of a pen event
#     current_id = 0
#     colors = [QColor(255, 255, 255, 255), QColor(0, 0, 255, 255), QColor(0, 255, 0, 255), QColor(255, 0, 0, 255),
#               QColor(0, 255, 255, 255), QColor(255, 0, 255, 255), QColor(255, 255, 0, 255)]
#     color_index = 0
#     current_color = QColor(255, 255, 255, 255)
#
#     start_time = time.time()
#     frame_counter = 0
#
#     def get_next_color(self):
#         if self.color_index == len(self.colors):
#             self.color_index = 0
#
#         self.current_color = self.colors[self.color_index]
#         self.color_index += 1
#
#     def __init__(self, painting_widget):
#         QThread.__init__(self)
#         self.painting_widget = painting_widget
#
#     def __del__(self):
#         self.wait()
#
#     def run(self):
#         while True:
#             self.process_frames()
#
#             # if (time.time() - self.start_time) > 1:  # displays the frame rate every 1 second
#             #     print("FPS: %s" % round(self.frame_counter / (time.time() - self.start_time), 1))
#             #     self.frame_counter = 0
#             #     self.start_time = time.time()
#
#     # @timeit("Process frames")
#     def process_frames(self):
#         # global realsense_d435_camera
#         global flir_blackfly_s
#         # global ir_pen
#         # left_ir_image_1, left_ir_image_2, matrix1, matrix2 = realsense_d435_camera.get_camera_frames()
#
#         new_frames, matrices = flir_blackfly_s.get_camera_frames()
#
#         if len(new_frames) > 0:
#             self.frame_counter += 1
#
#             _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(new_frames[0])
#             if brightest > 100:
#                 # print('Bright!')
#                 self.painting_widget.fill_screen_white()
#             else:
#                 self.painting_widget.fill_screen_black()
#
#             return
#
#             # old: 12 - 15 ms
#             # crop: 9 - 10 ms
#             # start_time = time.time()
#
#             # pen_event_roi, brightest, (x, y) = ir_pen_1.crop_image(left_ir_image_1)
#             # coords = [x, y, 1]
#             # coords = np.array(coords)
#             #
#             # print(matrix1)
#             # result = matrix1.dot(coords)
#             # # result = matrix1 @ coords
#             # print(coords, result)
#
#             # active_pen_events, stored_lines, _, _, debug_distances = ir_pen.get_ir_pen_events_multicam([left_ir_image_1, left_ir_image_2], [matrix1, matrix2])
#             active_pen_events, stored_lines, _, _, debug_distances = ir_pen.get_ir_pen_events_multicam(new_frames, matrices)
#
#             # active_pen_events, stored_lines, _, _, debug_distances = [], [], [], [], []
#             global current_debug_distances
#             current_debug_distances = debug_distances
#             # end_time = time.time()
#             # print('get_ir_pen_events_multicam: {} ms'.format(round((end_time - start_time) * 1000, 2)))
#
#
#
#             # start_time = time.time()
#             # end_time = time.time()
#             # print('findContours', round((end_time - start_time) * 1000, 2))
#
#
#             # zeroes = np.zeros(projection_area_frames[0].shape, 'uint8')
#             # ir_image_fake_color = np.dstack((projection_area_frames[0], projection_area_frames[1], zeroes))
#
#             # ir_image_fake_color_extended = np.dstack((current_ir_image_table_0, current_ir_image_table_1, zeroes))
#             #
#             # img_pen_roi, brightest, _ = ir_pen_1.crop_image(ir_image_fake_color_extended)
#             #
#             # if brightest < 100:
#             #     img_pen_roi = np.zeros((48, 48), 'uint8')
#             #
#             # img_pen_roi = cv2.resize(img_pen_roi, (1024, 1024))
#
#             self.painting_widget.draw_all_points(active_pen_events, stored_lines) # 1.2 ms
#         else:
#             # We need this sleep. Otherwise we will overload the camera script with too many requests for new frames.
#             # TODO: Find a better solution
#             time.sleep(0.0001)
#
#             # end_time = time.time()
#             # run_time = (end_time - self.start_time) * 1000
#             # print(run_time, flush=True)
#             # self.start_time = time.time()
#
#             # self.painting_widget.preview_images_on_canvas(
#             #     [current_ir_image_table_1, current_ir_image_table_2, ir_image_fake_color, img_pen_roi],
#             #     ['right ir', 'left ir', 'Fake color', 'ROI'])
#
#     # def process_frame_stereo(self):
#     #     global realsense_d435_camera
#     #     global ir_pen_1
#     #     global ir_pen_2
#     #
#     #     ir_image_table, current_ir_image_table_1, current_ir_image_table_2 = realsense_d435_camera.get_ir_image()
#     #
#     #     if ir_image_table is not None:
#     #         active_pen_events, _, _, pen_events_to_remove = ir_pen_1.get_ir_pen_events(ir_image_table)
#     #
#     #         if len(pen_events_to_remove) > 0:
#     #             self.painting_widget.reset_last_point()
#     #
#     #         new_points = []
#     #         for active_pen_event in active_pen_events:
#     #
#     #             if COLOR_CYCLE_TESTING:
#     #                 if active_pen_event.id > self.current_id:
#     #                     self.get_next_color()
#     #                     self.current_id = active_pen_event.id
#     #
#     #             if active_pen_event.state.value != 3:  # All events except hover
#     #                 if len(active_pen_event.history) > NUM_POINTS_IGNORE:
#     #                         new_points.append(QPoint(active_pen_event.x, active_pen_event.y))
#     #
#     #         # for active_pen_event in active_pen_events_2:
#     #         #     if active_pen_event.state.value != 3:  # All events except hover
#     #         #         if len(active_pen_event.history) > NUM_POINTS_IGNORE:
#     #         #             new_points.append(QPoint(active_pen_event.x, active_pen_event.y))
#     #
#     #         if COLOR_CYCLE_TESTING:
#     #             self.painting_widget.draw_new_points(new_points, self.current_color)
#     #         else:
#     #             self.painting_widget.draw_new_points(new_points)
#
#     # @timeit("Process Frame")
#     # def process_frame(self):
#     #     global realsense_d435_camera
#     #     global ir_pen_1
#     #     global ir_pen_2
#     #     current_ir_image_table_1, current_ir_image_table_2, left_ir_image_1, left_ir_image_2, crop_1, crop_2 = realsense_d435_camera.get_ir_image()
#     #
#     #     if current_ir_image_table_1 is not None and current_ir_image_table_2 is not None:
#     #
#     #
#     #         current_ir_image_table_cropped_1 = current_ir_image_table_1[crop_1[1] : crop_1[3], crop_1[0]: crop_1[2]]
#     #         current_ir_image_table_cropped_2 = current_ir_image_table_2[crop_2[1] : crop_2[3], crop_2[0]: crop_2[2]]
#     #         current_ir_image_table_cropped_1 = cv2.resize(current_ir_image_table_cropped_1, (848, 480))
#     #         current_ir_image_table_cropped_2 = cv2.resize(current_ir_image_table_cropped_2, (848, 480))
#     #
#     #         zeroes = np.zeros(current_ir_image_table_cropped_1.shape, 'uint8')
#     #         ir_image_fake_color = np.dstack((current_ir_image_table_cropped_1, current_ir_image_table_cropped_2, zeroes))
#     #
#     #         #print(crop_1, current_ir_image_table_1.shape, current_ir_image_table_cropped_1.shape)
#     #
#     #         pos = None
#     #         text = ''
#     #
#     #         if STEREO_MODE:
#     #             active_pen_events, _, _, pen_events_to_remove, data_1 = ir_pen_1.get_ir_pen_events(
#     #                 ir_image_fake_color)
#     #         else:
#     #
#     #             _, brightest_1, _, (max_x_1, max_y_1) = cv2.minMaxLoc(current_ir_image_table_cropped_1)
#     #             _, brightest_2, _, (max_x_2, max_y_2) = cv2.minMaxLoc(current_ir_image_table_cropped_2)
#     #
#     #             # LATENZ MESSUNG
#     #             # if brightest_1 > 100:
#     #             #     self.painting_widget.fill_screen(True)
#     #             #     return
#     #             # else:
#     #             #     self.painting_widget.fill_screen(False)
#     #             #     return
#     #
#     #
#     #             MIN_BRIGHTNESS = 60
#     #
#     #
#     #
#     #             active_pen_events_1, _, _, pen_events_to_remove_1, data_1 = ir_pen_1.get_ir_pen_events(
#     #                 current_ir_image_table_cropped_1)
#     #             active_pen_events_2, _, _, pen_events_to_remove_2, data_2 = ir_pen_2.get_ir_pen_events(
#     #                 current_ir_image_table_cropped_2)
#     #
#     #             dist = 0
#     #
#     #             if len(active_pen_events_1) > 0 and len(active_pen_events_2) > 0:
#     #                 dist = distance.euclidean((active_pen_events_1[0].x, active_pen_events_1[0].y),
#     #                                           (active_pen_events_2[0].x, active_pen_events_2[0].y))
#     #                 dist = int(dist)
#     #
#     #             # print('DIST:', dist)
#     #
#     #             # print(data_1, data_2)
#     #
#     #             active_pen_events = []
#     #             pen_events_to_remove = pen_events_to_remove_1 + pen_events_to_remove_2
#     #
#     #             # _, thresh_1 = cv2.threshold(current_ir_image_table_1, int(np.median(current_ir_image_table_1) * 1.5), 255, cv2.THRESH_BINARY)
#     #             # _, thresh_2 = cv2.threshold(current_ir_image_table_2, int(np.median(current_ir_image_table_2) * 1.5), 255, cv2.THRESH_BINARY)
#     #
#     #             value_offset = 0.5
#     #             _, thresh_1 = cv2.threshold(current_ir_image_table_cropped_1, (brightest_1 * value_offset) if brightest_1 > MIN_BRIGHTNESS else 254,  255,
#     #                                         cv2.THRESH_BINARY)
#     #             _, thresh_2 = cv2.threshold(current_ir_image_table_cropped_2, (brightest_2 * value_offset) if brightest_2 > MIN_BRIGHTNESS else 254, 255, cv2.THRESH_BINARY)
#     #
#     #             dest_and = cv2.bitwise_and(thresh_1, thresh_2, mask=None)
#     #
#     #             # self.painting_widget.preview_images_on_canvas([current_ir_image_table_1, current_ir_image_table_2, ir_image_fake_color, dest_and],
#     #             #self.painting_widget.preview_images_on_canvas([current_ir_image_table_1, current_ir_image_table_2, ir_image_fake_color, dest_and],
#     #             #                                              ['right ir', 'left ir', 'Fake color', 'AND'])
#     #
#     #             max_and = np.max(dest_and)
#     #             max_thresh_1 = np.max(thresh_1)
#     #             max_thresh_2 = np.max(thresh_2)
#     #
#     #             # print(brightest_1, brightest_2, np.max(dest_and), np.max(thresh_1), np.max(thresh_2))
#     #
#     #             # if (brightest_1 < MIN_BRIGHTNESS and brightest_2 < MIN_BRIGHTNESS) or (np.max(thresh_1) == 0 and np.max(thresh_2) == 0):
#     #             #
#     #             # # elif brightest_1 >= MIN_BRIGHTNESS and brightest_2 < MIN_BRIGHTNESS:
#     #
#     #             if max_and != 0:
#     #             # if 0 < dist < 50:
#     #                 non_zero_px = cv2.countNonZero(dest_and)
#     #
#     #                 success, new_pos = self.find_pen_position_subpixel(dest_and)
#     #                 if success:
#     #                     pos = new_pos
#     #
#     #                 if brightest_1 > brightest_2:
#     #                     text = 'both - using cam 1 (non zero px: {})'.format(non_zero_px)
#     #                     active_pen_events = active_pen_events_1
#     #                     data = data_1
#     #                 else:
#     #                     text = 'both - using cam 2 (non zero px: {})'.format(non_zero_px)
#     #                     active_pen_events = active_pen_events_2
#     #                     data = data_2
#     #             elif max_thresh_1 == 0 and max_thresh_2 == 0:
#     #                 text = 'No Pen'
#     #
#     #             elif max_thresh_1 != 0 and max_thresh_2 != 0:
#     #                 text = 'hover far'
#     #                 # cv2.imwrite('additional_training_images/false_hover_far/hover_{}.png'.format(round(time.time() * 1000)),
#     #                 #             ir_image_fake_color)
#     #
#     #             elif brightest_1 >= MIN_BRIGHTNESS and max_thresh_2 == 0:
#     #                 text = 'cam right'
#     #                 active_pen_events = active_pen_events_1
#     #                 data = data_1
#     #
#     #             # elif brightest_2 >= MIN_BRIGHTNESS and brightest_1 < MIN_BRIGHTNESS:
#     #             elif brightest_2 >= MIN_BRIGHTNESS and max_thresh_1 == 0:
#     #                 text = 'cam left'
#     #                 active_pen_events = active_pen_events_2
#     #                 data = data_2
#     #             else:
#     #                 text = 'We missed something'
#     #
#     #         if len(pen_events_to_remove) > 0:
#     #             self.painting_widget.reset_last_point()
#     #
#     #             # self.painting_widget.add_data(data)
#     #
#     #         new_points = []
#     #         # if 40 > dist > 0:
#     #         for active_pen_event in active_pen_events:
#     #
#     #             if COLOR_CYCLE_TESTING:
#     #                 if active_pen_event.id > self.current_id:
#     #                     self.get_next_color()
#     #                     self.current_id = active_pen_event.id
#     #
#     #             #if active_pen_event.state.value != 3:
#     #             #    cv2.imwrite('additional_training_images/hover/hover_{}.png'.format(round(time.time() * 1000)), ir_image_table)
#     #             #
#     #             # print(active_pen_event)
#     #             if active_pen_event.state.value != 3:  # All events except hover
#     #                 if len(active_pen_event.history) > NUM_POINTS_IGNORE:
#     #                     if pos is not None:
#     #                         new_points.append(QPoint(pos[0], pos[1]))
#     #                     else:
#     #                         new_points.append(QPoint(active_pen_event.x, active_pen_event.y))
#     #
#     #
#     #
#     #         # for active_pen_event in active_pen_events_2:
#     #         #     if active_pen_event.state.value != 3:  # All events except hover
#     #         #         if len(active_pen_event.history) > NUM_POINTS_IGNORE:
#     #         #             new_points.append(QPoint(active_pen_event.x, active_pen_event.y))
#     #
#     #         text = '{}; Dist: {}'.format(brightest_1 - brightest_2, dist)+ ' ' + text
#     #
#     #         if COLOR_CYCLE_TESTING:
#     #             self.painting_widget.draw_new_points(new_points, text, self.current_color)
#     #         else:
#     #             self.painting_widget.draw_new_points(new_points, text)


# class PaintingWidget(QMainWindow):
#
#     # TODO: Change this to work with multiple pens at once
#     # lastPoint = None
#
#     mackenzie_phrases = []
#
#     # height_multiplier = 2
#
#     num_phrases_written = 0
#
#     stored_data = []
#
#     def __init__(self):
#         super().__init__()
#
#         # shape_creator = ShapeCreator(WINDOW_WIDTH, WINDOW_HEIGHT)
#         # background_image = shape_creator.draw_shape('shapes/wave.svg', (800, 800), 1000, LINE_THICKNESS, ShapeCreator.DASH, LINE_COLOR)
#         # self.background_image = QImage(background_image, background_image.shape[1], background_image.shape[0], background_image.shape[1] * 3, QImage.Format_RGB888)
#
#         self.read_mackenzie_phrases()
#
#         # drawing flag
#         self.drawing = False
#
#         self.line_thickness = LINE_THICKNESS
#
#         self.initUI()
#
#         self.thread = ApplicationLoopThread(self)
#         self.thread.start()
#
#     def add_data(self, data):
#         print(len(self.stored_data))
#
#         if len(data.keys()) > 0:
#             self.stored_data.append(data)
#
#     def initUI(self):
#         self.showFullScreen()  # Application should run in Fullscreen
#
#         # setting geometry to main window
#         self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
#
#         # Define width and height as global variables
#         # self.width = QApplication.desktop().screenGeometry().width()
#         # self.height = QApplication.desktop().screenGeometry().height()
#
#         # hover_cursor = QLabel(self)
#         # pixmap = QPixmap("testing.png")
#         # # pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
#         # hover_cursor.setPixmap(pixmap)
#         # hover_cursor.move(100, 100)
#         #
#         # label = QLabel(text="Welcome to Python GUI!")
#         # label.show()
#
#         # self.reset_image()
#
#     def read_mackenzie_phrases(self):
#         with open('phrases.txt') as file:
#             lines = file.readlines()
#             for line in lines:
#                 self.mackenzie_phrases.append(line.replace('\n', ''))
#
#     def get_random_phrase(self):
#
#         index = random.randint(0, len(self.mackenzie_phrases) - 1)
#
#         random_phrase = self.mackenzie_phrases.pop(index)
#
#         return random_phrase
#
#
#     def reset_image(self):
#         # self.image = self.background_image.copy()
#         self.image = QImage(self.size(), QImage.Format_ARGB32)
#         self.image.fill(Qt.black)
#
#         if PHRASES_MODE:
#             self.draw_rectangle()
#
#     # Only for latency measurements
#     def fill_screen(self, fill):
#         self.image = QImage(self.size(), QImage.Format_ARGB32)
#         if fill:
#             self.image.fill(Qt.white)
#         else:
#             self.image.fill(Qt.black)
#         self.update()
#
#     def draw_all_points(self, active_pen_events, stored_lines):
#
#         # self.reset_image()
#
#         new_image = QImage(self.size(), QImage.Format_ARGB32)
#         new_image.fill(Qt.black)
#
#         painter = QPainter(new_image)
#
#         color = QColor(255, 255, 255, 255)
#         color_hover = QColor(255, 255, 255, 40)
#
#         painter.setPen(QPen(color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#
#         global last_time
#         font = QFont()
#         font.setFamily('Arial')
#         # font.setBold(True)
#         font.setPointSize(40)
#         painter.setFont(font)
#
#         painter.drawText(100, 550, str(int(last_time)) + ' ms')
#
#         polygons_to_draw = []
#
#         for active_pen_event in active_pen_events:
#
#             polygon = []
#
#             if active_pen_event.state.value != 3:  # All events except hover
#                 for point in active_pen_event.history:
#                     polygon.append(QPoint(point[0], point[1]))
#             else:
#                 # Draw a dot to show hover events
#                 painter.setPen(QPen(color_hover, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#                 painter.setBrush(QBrush(color_hover, Qt.SolidPattern))
#                 painter.drawEllipse(active_pen_event.x, active_pen_event.y, 5, 5)
#
#                 painter.setPen(QPen(color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#
#             if len(polygon) > 0:
#                 polygons_to_draw.append(polygon)
#
#         for line in stored_lines:
#
#             polygon = []
#
#             for point in line:
#                 polygon.append(QPoint(point[0], point[1]))
#
#             if len(polygon) > 0:
#                 polygons_to_draw.append(polygon)
#
#         for polygon in polygons_to_draw:
#             # painter.drawPolygon(QPolygon(polygon))
#             painter.drawPolyline(QPolygon(polygon))
#
#         self.image = new_image
#         self.update()
#
#
#     # def draw_new_points(self, points, text, color=PEN_COLOR):
#     #
#     #     if 'left' in text:
#     #         color = QColor(255, 0, 255, 255)
#     #     elif 'right' in text:
#     #         color = QColor(255, 255, 0, 255)
#     #     elif 'cam 1' in text:
#     #         color = QColor(0, 255, 0, 255)
#     #     elif 'cam 2' in text:
#     #         color = QColor(255, 0, 0, 255)
#     #
#     #
#     #
#     #     # print('drawing', points)
#     #
#     #     painter = QPainter(self.image)
#     #     # painter.setRenderHint(QPainter.Antialiasing)
#     #
#     #     RECTANGLE_COLOR = QColor(40, 40, 40, 255)
#     #
#     #     # painter.setRenderHint(QPainter.Antialiasing)
#     #     painter.setPen(QPen(RECTANGLE_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#     #     painter.setBrush(QBrush(RECTANGLE_COLOR))
#     #     painter.drawRect(80, 500, 1000, 100)
#     #
#     #     painter.setPen(QPen(color, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#     #
#     #     font = QFont()
#     #     font.setFamily('Arial')
#     #     # font.setBold(True)
#     #     font.setPointSize(40)
#     #     painter.setFont(font)
#     #
#     #     painter.drawText(100, 550, text)
#     #
#     #     for point in points:
#     #         if self.lastPoint is None:
#     #             painter.drawPoint(point)
#     #         else:
#     #             painter.drawLine(self.lastPoint, point)
#     #         self.lastPoint = point
#     #
#     #     self.update()
#     #
#     # def draw_rectangle(self):
#     #     painter = QPainter(self.image)
#     #
#     #     painter.setPen(QPen(Qt.white))
#     #
#     #     font = QFont()
#     #     font.setFamily('Arial')
#     #     # font.setBold(True)
#     #     font.setPointSize(80)
#     #     painter.setFont(font)
#     #
#     #     painter.drawText(100, 300, self.get_random_phrase())
#     #
#     #     RECTANGLE_COLOR = QColor(50, 50, 50, 255)
#     #
#     #     # painter.setRenderHint(QPainter.Antialiasing)
#     #     painter.setPen(QPen(RECTANGLE_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#     #     painter.setBrush(QBrush(RECTANGLE_COLOR))
#     #
#     #     if self.height_multiplier > 3:
#     #         self.height_multiplier = 2
#     #
#     #     height = 58 * self.height_multiplier
#     #     width = WINDOW_WIDTH/2
#     #
#     #     self.height_multiplier += 1
#     #
#     #     painter.drawRect(int(WINDOW_WIDTH/2 - width/2), int(WINDOW_HEIGHT/2 - height/2), width, height)
#     #
#     #     self.update()
#     #
#     # def reset_last_point(self):
#     #     print('##########################################')
#     #     self.lastPoint = None
#
#     # def mousePressEvent(self, event):
#     #     if event.button() == Qt.LeftButton:
#     #         self.drawing = True
#     #         self.lastPoint = event.pos()
#     #
#     # def mouseMoveEvent(self, event):
#     #     # checking if left button is pressed and drawing flag is true
#     #     if (event.buttons() & Qt.LeftButton) & self.drawing:
#     #         # creating painter object
#     #         painter = QPainter(self.image)
#     #
#     #         # set the pen of the painter
#     #         painter.setPen(QPen(PEN_COLOR, self.line_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#     #
#     #         # draw line from the last point of cursor to the current point
#     #         # this will draw only one step
#     #         # painter.drawLine(self.lastPoint, event.pos())
#     #         painter.drawPoint(event.pos())
#     #
#     #         self.lastPoint = event.pos()
#     #         self.update()
#     #
#     # def mouseReleaseEvent(self, event):
#     #     if event.button() == Qt.LeftButton:
#     #         self.drawing = False
#
#     # paint event
#     def paintEvent(self, event):
#         # create a canvas
#         canvasPainter = QPainter(self)
#
#         # draw rectangle  on the canvas
#         canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
#
#     # Handle Key-press events
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Escape:
#             # df = pd.DataFrame(self.stored_data)
#             # df.to_csv('stored_data_hover.csv')
#
#             self.save_screenshot()
#
#             self.close()
#             sys.exit(0)
#         elif event.key() == Qt.Key_Space:
#             self.save_screenshot()
#
#             if PHRASES_MODE:
#                 self.num_phrases_written += 1
#                 if self.num_phrases_written == 10:
#                     print('10 Phrases have been written')
#                     self.close()
#                     sys.exit()
#
#             self.reset_image()
#
#     # Save the current window content as a png
#     def save_screenshot(self):
#         filename = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
#         self.image.save('./output_images/Participant_{}_{}.png'.format(PARTICIPANT_ID, filename))
#
#     def preview_images_on_canvas(self, images_to_preview, image_names):
#         painter = QPainter(self.image)
#
#         x_anchor = 3840
#
#         for i, image_to_preview in enumerate(images_to_preview):
#             if len(image_to_preview.shape) < 3:
#                 image_to_preview = cv2.cvtColor(image_to_preview, cv2.COLOR_GRAY2RGB)
#             image_to_preview = cv2.putText(
#                 img=image_to_preview,
#                 text=image_names[i],
#                 org=(50, 50),
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=2.0,
#                 color=(255, 255, 255),
#                 thickness=2
#             )
#             image_to_preview = cv2.copyMakeBorder(image_to_preview, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,255,255))
#             preview_image = QImage(image_to_preview.data, image_to_preview.shape[1], image_to_preview.shape[0],
#                                    QImage.Format_RGB888)
#
#             x_anchor -= image_to_preview.shape[1]
#
#             painter.drawImage(x_anchor, 0, preview_image)
#
#         self.update()



