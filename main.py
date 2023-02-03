
import os
import sys
import cv2
import socket
import threading
import numpy as np

from pen_state import PenState
from ir_pen import IRPen
from flir_blackfly_s import FlirBlackflyS

ENABLE_FIFO_PIPE = False
ENABLE_UNIX_SOCKET = True
UNIX_SOCK_NAME = 'uds_test'
PIPE_NAME = 'pipe_test'

DEBUG_MODE = 0  # Enable for Debug print statements and preview windows
SEND_TO_FRONTEND = 1  # Enable if points should be forwarded to the sdl frontend

TRAINING_DATA_COLLECTION_MODE = False  # Enable if ROIs should be saved to disk

DOCUMENTS_DEMO = False

if DOCUMENTS_DEMO:
    from logitech_brio import LogitechBrio
    from AnalogueDigitalDocumentsDemo import AnalogueDigitalDocumentsDemo


class Main:

    rois = []  # Regions of interest

    uds_initialized = False
    sock = None
    pipeout = None

    last_heartbeat_timestamp = 0

    preview_initialized = False

    def __init__(self):

        if SEND_TO_FRONTEND:
            if ENABLE_FIFO_PIPE:
                if not os.path.exists(PIPE_NAME):
                    os.mkfifo(PIPE_NAME)
                self.pipeout = os.open(PIPE_NAME, os.O_WRONLY)

            if ENABLE_UNIX_SOCKET:
                self.init_unix_socket()

        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        if TRAINING_DATA_COLLECTION_MODE:
            from training_images_collector import TrainingImagesCollector
            self.training_images_collector = TrainingImagesCollector(self.ir_pen, self.flir_blackfly_s.EXPOSURE_TIME_MICROSECONDS, self.flir_blackfly_s.GAIN)

        if DOCUMENTS_DEMO:
            self.analogue_digital_document = AnalogueDigitalDocumentsDemo()

            self.logitech_brio_camera = LogitechBrio(self)
            self.logitech_brio_camera.init_video_capture()
            self.logitech_brio_camera.start()

        thread = threading.Thread(target=self.main_thread)
        thread.start()

    def init_unix_socket(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        print('Connecting to UNIX Socket %s' % UNIX_SOCK_NAME)
        try:
            self.sock.connect(UNIX_SOCK_NAME)
        except socket.error as error:
            print('Error while connecting to UNIX Socket:', error)
            print('Make sure that the frontend is already running before starting this python script')
            sys.exit(1)
        self.uds_initialized = True

    # ----------------------------------------------------------------------------------------------------------------

    # For documents demo
    def on_new_brio_frame(self, frame, homography_matrix):
        # print(frame.shape)

        document_found, highlight_dict, document_changed, document_removed, document_moved, \
            converted_document_corner_points, document_moved_matrix = self.analogue_digital_document.get_highlight_rectangles(frame, homography_matrix)

        self.send_heartbeat(document_found)

        if document_moved and not document_removed:
            self.send_corner_points(converted_document_corner_points)  # , int(not document_removed) Andi was here
            try:
                for highlight_id, rectangle in highlight_dict.items():
                    self.send_rect(highlight_id, 1, rectangle)
            except:
                print('error sending highlights')

        if document_removed:
            self.send_corner_points([])

        if document_found:
            if len(document_moved_matrix) > 0:
                self.send_matrix(document_moved_matrix)

        if document_changed or document_removed:
            self.clear_rects()

    # For documents demo
    def send_heartbeat(self, document_found):
        message = f's {int(document_found)} |'

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            try:
                # print('SEND HEARTBEAT')
                self.sock.sendall(message.encode())
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in send_heartbeat')
                self.init_unix_socket()

        return 1

    # For documents demo
    def send_matrix(self, matrix):
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        flat = [item for sublist in matrix for item in sublist]

        message = f'm '

        for i in flat:
            message += f'{i} '

        message += '|'

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            try:
                # print('SEND MATRIX')
                self.sock.sendall(message.encode())
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in send_matrix')
                self.init_unix_socket()

        return 1

    # For documents demo
    def send_corner_points(self, converted_document_corner_points):
        if not self.uds_initialized:
            return 0

        message = f'k '
        if len(converted_document_corner_points) == 4:

            for point in converted_document_corner_points:
                message += f'{int(point[0])} {int(point[1])} '
            message += '1 '  # document exists
        else:

            for i in range(9):
                message += '0 '

        message += '|'

        # print('message', message)

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            # print('SEND CORNER POINTS')
            try:
                self.sock.sendall(message.encode())
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in send_corner_points')
                self.init_unix_socket()

        return 1

    # For documents demo
    # id: id of the rect (writing to an existing id should move the rect)
    # state: alive = 1, dead = 0; use it to remove unused rects!
    # coords list: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] -- use exactly four entries and make sure they are sorted!
    def send_rect(self, rect_id, state, coord_list):
        if not self.uds_initialized:
            return 0

        message = f'r {rect_id} '
        for coords in coord_list:
            # message += f'{coords[0]} {coords[1]}'
            message += f'{coords} '
        message += f'{state} '

        message += '|'

        # print('message', message)

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            try:
                # print('SEND RECT')
                self.sock.sendall(message.encode())
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in send_rect()')
                self.init_unix_socket()

        return 1

    # For documents demo
    def delete_line(self, line_id):
        if not self.uds_initialized:
            return 0

        message = f'd {line_id} |'

        print('DELETING LINE WITH ID', line_id)
        print('message:', message)

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            try:
                self.sock.sendall(message.encode())
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in delete_line()')
                self.init_unix_socket()

    # For documents demo
    def clear_rects(self):
        if not self.uds_initialized:
            return 0

        if ENABLE_UNIX_SOCKET:
            try:
                print('CLEAR RECTS')
                self.sock.sendall('c |'.encode())
                return 1
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in clear_rects()')
                self.init_unix_socket()

    # ----------------------------------------------------------------------------------------------------------------

    def add_new_line_point(self, active_pen_event):
        if not self.uds_initialized:
            print('Error in add_new_line_point(): uds not initialized')
            return False

        message = 'l {} 255 255 255 {} {} {} |'.format(active_pen_event.id,
                                                       int(active_pen_event.x),
                                                       int(active_pen_event.y),
                                                       0 if active_pen_event.state == PenState.HOVER else 1)

        if ENABLE_FIFO_PIPE:
            os.write(self.pipeout, bytes(message, 'utf8'))
        if ENABLE_UNIX_SOCKET:
            try:
                self.sock.sendall(message.encode())
            except Exception as e:
                print('---------')
                print(e)
                print('Broken Pipe in add_new_line_point()')
                self.init_unix_socket()

    # def append_line(self, line_id, line):
    #
    #     print('append', line_id, line)
    #
    #     if not self.uds_initialized:
    #         return 0
    #
    #     line_color = '255 255 255'
    #
    #     message = f'a {line_id} {line_color} '
    #
    #     for i, point in enumerate(line):
    #         message += f'{int(point[0])},{int(point[1])}'
    #         if i < len(line) - 1:
    #             message += ';'
    #
    #     message += ' - |'  # Append message end
    #
    #     print('message append', len(message), message)
    #
    #     if ENABLE_FIFO_PIPE:
    #         os.write(self.pipeout, bytes(message, 'utf8'))
    #     if ENABLE_UNIX_SOCKET:
    #         try:
    #             self.sock.send(message.encode())
    #         except Exception as e:
    #             print(e)
    #             print('---------')
    #             print('Broken Pipe in append_line')
    #             self.init_unix_socket()

    def main_thread(self):
        while True:

            if DEBUG_MODE:
                self.show_debug_preview()

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    sys.exit(0)
            else:
                # TODO: Check why we need a delay here. Without it, it will lag horribly.
                #  time.sleep() does not work here
                cv2.waitKey(1)

    def show_debug_preview(self):

        # TODO: Label origin camera

        if not self.preview_initialized:
            cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ROI', 1920, 960)
            self.preview_initialized = True

        if len(self.rois) == 2:
            roi0 = cv2.resize(self.rois[0], (960, 960), interpolation=cv2.INTER_AREA)
            roi1 = cv2.resize(self.rois[1], (960, 960), interpolation=cv2.INTER_AREA)

            roi0 = self.add_max_brightness_label(roi0, np.max(roi0))
            roi1 = self.add_max_brightness_label(roi1, np.max(roi1))

            cv2.imshow('ROI', cv2.hconcat([roi0, roi1]))

        elif len(self.rois) == 1:
            roi0 = cv2.resize(self.rois[0], (960, 960), interpolation=cv2.INTER_AREA)
            roi1 = np.zeros((960, 960), np.uint8)

            roi0 = self.add_max_brightness_label(roi0, np.max(roi0))

            cv2.imshow('ROI', cv2.hconcat([roi0, roi1]))

        self.rois = []

    def add_max_brightness_label(self, frame, max_brightness):
        return cv2.putText(
            img=frame,
            text=str(max_brightness),
            org=(90, 90),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=2.0,
            color=(255),
            thickness=3
        )

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(frames) > 0:
            if TRAINING_DATA_COLLECTION_MODE:
                self.training_images_collector.save_training_images(frames)
            else:
                active_pen_events, stored_lines, _, _, rois = self.ir_pen.get_ir_pen_events(frames, matrices)

                if SEND_TO_FRONTEND:
                    for active_pen_event in active_pen_events:
                        self.add_new_line_point(active_pen_event)

                self.rois = rois

                if DOCUMENTS_DEMO:
                    self.analogue_digital_document.on_new_finished_lines(stored_lines)


if __name__ == '__main__':
    main = Main()
