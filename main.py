
import os
import sys
import cv2
import socket
import threading

from pen_state import PenState
from ir_pen import IRPen
from flir_blackfly_s import FlirBlackflyS

ENABLE_FIFO_PIPE = False
ENABLE_UNIX_SOCKET = True
UNIX_SOCK_NAME = 'uds_test'
PIPE_NAME = 'pipe_test'

DEBUG_MODE = 0  # Enable for Debug print statements and preview windows
SEND_TO_FRONTEND = True  # Enable if points should be forwarded to the sdl frontend

TRAINING_DATA_COLLECTION_MODE = False  # Enable if ROIs should be saved to disk

DOCUMENTS_DEMO = False

if DOCUMENTS_DEMO:
    from logitech_brio import LogitechBrio
    from AnalogueDigitalDocumentsDemo import AnalogueDigitalDocumentsDemo


class Main:

    uds_initialized = False
    socket = None
    pipeout = None

    last_heartbeat_timestamp = 0

    preview_initialized = False

    def __init__(self):

        if SEND_TO_FRONTEND:
            if ENABLE_FIFO_PIPE:
                self.init_fifo_pipe()
            if ENABLE_UNIX_SOCKET:
                self.init_unix_socket()

        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        if TRAINING_DATA_COLLECTION_MODE:
            from training_images_collector import TrainingImagesCollector
            exposure = self.flir_blackfly_s.get_exposure_time()
            gain = self.flir_blackfly_s.get_gain()
            self.training_images_collector = TrainingImagesCollector(self.ir_pen, exposure, gain)

        if DOCUMENTS_DEMO:
            self.analogue_digital_document = AnalogueDigitalDocumentsDemo()

            self.logitech_brio_camera = LogitechBrio(self)
            self.logitech_brio_camera.init_video_capture()
            self.logitech_brio_camera.start()

        # Start a thread to keep this script alive
        thread = threading.Thread(target=self.main_thread)
        thread.start()

    def init_fifo_pipe(self):
        if not os.path.exists(PIPE_NAME):
            os.mkfifo(PIPE_NAME)
        self.pipeout = os.open(PIPE_NAME, os.O_WRONLY)

    def init_unix_socket(self):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        print('Connecting to UNIX Socket %s' % UNIX_SOCK_NAME)
        try:
            self.socket.connect(UNIX_SOCK_NAME)
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
        message = f's {int(document_found)}'

        self.send_message(message)

    # For documents demo
    def send_matrix(self, matrix):
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        flat = [item for sublist in matrix for item in sublist]

        message = f'm'

        for i in flat:
            message += f' {i}'

        self.send_message(message)

    # For documents demo
    def send_corner_points(self, converted_document_corner_points):
        if not self.uds_initialized:
            return 0

        message = f'k'
        if len(converted_document_corner_points) == 4:

            for point in converted_document_corner_points:
                message += f'{int(point[0])} {int(point[1])} '
            message += ' 1'  # document exists
        else:

            for i in range(9):
                message += ' 0'

        self.send_message(message)

    # For documents demo
    # id: id of the rect (writing to an existing id should move the rect)
    # state: alive = 1, dead = 0; use it to remove unused rects!
    # coords list: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] -- use exactly four entries and make sure they are sorted!
    def send_rect(self, rect_id, state, coord_list):

        message = f'r {rect_id}'
        for coords in coord_list:
            # message += f'{coords[0]} {coords[1]}'
            message += f' {coords}'
        message += f' {state}'

        self.send_message(message)

        return 1

    # For documents demo
    def delete_line(self, line_id):
        message = f'd {line_id}'

        print('DELETING LINE WITH ID', line_id)
        print('message:', message)

        self.send_message(message)

    # For documents demo
    def clear_rects(self):
        if not self.uds_initialized:
            return 0

        if ENABLE_UNIX_SOCKET:
            try:
                print('CLEAR RECTS')
                self.socket.sendall('c |'.encode())
                return 1
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in clear_rects()')
                self.init_unix_socket()

    # ----------------------------------------------------------------------------------------------------------------

    def finish_line(self, pen_event_to_remove):
        message = 'f {}'.format(pen_event_to_remove.id)

        print('Finish line', pen_event_to_remove.id)

        self.send_message(message)

    def add_new_line_point(self, active_pen_event):
        r = 255
        g = 255
        b = 255

        if active_pen_event.id % 3 == 0:
            r = 0
        if active_pen_event.id % 3 == 1:
            g = 0
        if active_pen_event.id % 3 == 2:
            b = 0

        message = 'l {} {} {} {} {} {} {}'.format(active_pen_event.id, r, g, b,
                                                       int(active_pen_event.x),
                                                       int(active_pen_event.y),
                                                       0 if active_pen_event.state == PenState.HOVER else 1)

        self.send_message(message)

    def send_message(self, message):
        if not self.uds_initialized:
            print('Error in finish_line(): uds not initialized')
        else:
            if ENABLE_FIFO_PIPE:
                # probably deprecated
                os.write(self.pipeout, bytes(message, 'utf8'))
            if ENABLE_UNIX_SOCKET:
                try:
                    msg_encoded = bytearray(message, 'ascii')
                    size = len(msg_encoded)
                    sock.send(size.to_bytes(4, 'big'))
                    sock.send(msg_encoded) # , MSG_NOSIGNAL
                except Exception as e:
                    print('---------')
                    print(e)
                    print('ERROR: Broken Pipe!')
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
            # TODO: Check why we need a delay here. Without it, it will lag horribly.
            #  time.sleep() does not work here
            cv2.waitKey(1)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):

        if len(frames) > 0:
            if TRAINING_DATA_COLLECTION_MODE:
                self.training_images_collector.save_training_images(frames)
            else:
                active_pen_events, stored_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events_new(frames, matrices)

                if SEND_TO_FRONTEND:
                    for active_pen_event in active_pen_events:
                        self.add_new_line_point(active_pen_event)

                    for pen_event in pen_events_to_remove:
                        self.finish_line(pen_event)

                if DOCUMENTS_DEMO:
                    self.analogue_digital_document.on_new_finished_lines(stored_lines)


if __name__ == '__main__':
    main = Main()
