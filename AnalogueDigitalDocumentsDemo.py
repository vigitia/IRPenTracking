import sys
import time

import cv2
import numpy as np

from DocumentLocatorService import DocumentLocatorService
from PDFAnnotationsService import PDFAnnotationsService
from FiducialsDetectorService import FiducialsDetectorService
from logitech_brio import LogitechBrio

# from scipy.spatial import distance

PDF_WIDTH = 595.446
PDF_HEIGHT = 841.691

DEBUG_MODE = False


class AnalogueDigitalDocumentsDemo:

    def __init__(self):

        self.document_locator_service = DocumentLocatorService()
        self.pdf_annotations_service = PDFAnnotationsService()
        self.fiducials_detection_service = FiducialsDetectorService()

        # self.logitech_brio_camera = LogitechBrio(self)
        # self.logitech_brio_camera.init_video_capture()
        # self.logitech_brio_camera.start()

    def get_highlight_rectangles(self, frame, transform_matrix):
        self.transform_matrix = transform_matrix

        highlights, notes, freehand_lines = self.pdf_annotations_service.get_annotations()

        # Detect ArUco markers
        aruco_markers = self.fiducials_detection_service.detect_fiducials(frame)

        # Locate the document. Order of document_corner_points TLC, BLC, BRC, TRC
        document_found, document_corner_points = self.locate_document(frame, aruco_markers)

        highlight_rectangles = []
        highlight_ids = []

        if document_found:
            frame, highlight_rectangles, highlight_ids = self.pdf_points_to_real_world(frame, highlights, document_corner_points)

            if DEBUG_MODE:
                frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]),
                                 (document_corner_points[2], document_corner_points[3]), (0, 0, 255), thickness=3)
                frame = cv2.line(frame, (document_corner_points[2], document_corner_points[3]),
                                 (document_corner_points[4], document_corner_points[5]), (0, 0, 255), thickness=3)
                frame = cv2.line(frame, (document_corner_points[4], document_corner_points[5]),
                                 (document_corner_points[6], document_corner_points[7]), (0, 0, 255), thickness=3)
                frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]),
                                 (document_corner_points[6], document_corner_points[7]), (0, 0, 255), thickness=3)

        if DEBUG_MODE:
            cv2.imshow('Logitech Brio', frame)
            cv2.waitKey(1)

        return highlight_rectangles, highlight_ids

    def on_new_brio_frame(self, frame, homography_matrix):
        self.get_highlight_rectangles(frame, [])

    def pdf_points_to_real_world(self, frame, highlights, document_corner_points):

        matrix = self.get_matrix_for_pdf_coordinate_transform(document_corner_points, to_pdf=False)

        highlight_ids = []
        all_highlight_points = []

        for highlight_object in highlights:
            highlight_group = highlight_object['quad_points']
            for i, highlight_list in enumerate(highlight_group):

                # Create a temporary ID for the highlight from the timestamp
                # TODO: Improve this!
                highlight_ids.append(int(highlight_object['timestamp'].replace('D:202211', '').replace("+01'00", '') + str(i)))
                highlight_points = self.list_to_points_list(highlight_list)
                all_highlight_points.append(highlight_points)

        # print(highlight_ids)
        # print(all_highlight_points)

        highlight_rectangles = []

        for highlight_group in all_highlight_points:
            points_to_be_transformed = np.array([highlight_group], dtype=np.float32)

            transformed_points = cv2.perspectiveTransform(points_to_be_transformed, matrix)
            transformed_points = transformed_points.tolist()[0]
            # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
            flat_list = [item for sublist in transformed_points for item in sublist]

            points_tuple_list = self.list_to_points_list(flat_list)

            # Transform to projection area in camera dimensions and then upscale to projector resolution
            points_tuple_list = self.transform_coords_to_output_res(points_tuple_list)
            flat_list = [item for sublist in points_tuple_list for item in sublist]


            # Convert floats to int
            for i, coordinate in enumerate(flat_list):
                flat_list[i] = int(coordinate)
            highlight_rectangles.append(flat_list)


            if DEBUG_MODE:
                cv2.fillPoly(frame, pts=[np.array(points_tuple_list)], color=(0, 255, 255))

            # print('points_list after transform', points_tuple_list)

        return frame, highlight_rectangles, highlight_ids

    def transform_coords_to_output_res(self, points_tuple_list):
        try:
            for i, point in enumerate(points_tuple_list):
                coords = np.array([point[0], point[1], 1])

                transformed_coords = self.transform_matrix.dot(coords)

                RES_2160P = True

                if RES_2160P:
                    # Normalize coordinates by dividing by z
                    # TODO: Improve this conversion. currently it only scales up from 1080p to 2160p by multiplying by 2
                    points_tuple_list[i] = ((transformed_coords[0] / transformed_coords[2]) * 2,
                                            (transformed_coords[1] / transformed_coords[2]) * 2)
                else:
                    # Normalize coordinates by dividing by z
                    points_tuple_list[i] = ((transformed_coords[0] / transformed_coords[2]),
                                            (transformed_coords[1] / transformed_coords[2]))

            return points_tuple_list
        except Exception as e:
            print(e)
            print('Error in transform_coords_to_output_res(). Maybe the transform_matrix is malformed?')
            print('This error could also appear if CALIBRATION_MODE is still enabled in flir_blackfly_s.py')
            time.sleep(5)
            sys.exit(1)

    def list_to_points_list(self, list_of_x_y_coords):
        points_list = []

        points_x = list_of_x_y_coords[::2]
        points_y = list_of_x_y_coords[1::2]

        for i in range(len(points_x)):
            point = (int(points_x[i]), int(points_y[i]))
            points_list.append(point)

        return points_list

    # def transform_pdf_point_to_real_world_document(self, pdf_x, pdf_y, document_corner_points):
    #
    #     # Calculate distance between top left corner and top right corner
    #     doc_width = distance.euclidean((document_corner_points[0], document_corner_points[1]),
    #                                    (document_corner_points[6], document_corner_points[7]))
    #
    #     # Calculate distance between top left corner and bottom left corner
    #     doc_height = distance.euclidean((document_corner_points[0], document_corner_points[1]),
    #                                     (document_corner_points[2], document_corner_points[3]))
    #
    #     width_fraction = doc_width / PDF_WIDTH
    #     height_fraction = doc_height / PDF_HEIGHT
    #
    #     x_new = pdf_x * width_fraction
    #     y_new = pdf_y * height_fraction
    #
    #     x_axis_normalized = None
    #     y_axis_normalized = None

    def on_new_finished_line(self, lines):
        if len(lines) > 0:
            print('Received stored lines:', len(lines))

    # Check if a point is on which side of a line. Return True if on right side, return False if on left side
    # https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
    def is_on_right_side(self, x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a * x0 - b * y0
        return a * x + b * y + c >= 0

    # This function will check if a point is inside the borders of the document
    # https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
    def is_point_on_document(self, x, y, document_corners):
        num_vert = len(document_corners)
        is_right = [self.is_on_right_side(x, y, document_corners[i], document_corners[(i + 1) % num_vert]) for i in
                    range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right

    def get_matrix_for_pdf_coordinate_transform(self, document_corner_points, to_pdf):

        document_corners = np.float32([[document_corner_points[0], document_corner_points[1]],
                                       [document_corner_points[2], document_corner_points[3]],
                                       [document_corner_points[4], document_corner_points[5]],
                                       [document_corner_points[6], document_corner_points[7]]])

        pdf_res = np.float32([[0, 0], [0, PDF_HEIGHT], [PDF_WIDTH, PDF_HEIGHT], [PDF_WIDTH, 0]])

        if to_pdf:
            matrix = cv2.getPerspectiveTransform(document_corners, pdf_res)
        else:
            matrix = cv2.getPerspectiveTransform(pdf_res, document_corners)

        # This matrix will be used to transform the points into the correct coordinate space for the pdf
        # (with origin in the bottom left corner and the correct resolution of the pdf)

        return matrix

    def add_lines_to_pdf(self, lines, document_corner_points):

        if len(lines) > 0 and len(document_corner_points) > 0:
            print('len(lines):', len(lines))
            # TODO: Rework
            print('add lines to pdf')
            document_corner_tuples = [(document_corner_points[0], document_corner_points[1]),
                                      (document_corner_points[2], document_corner_points[3]),
                                      (document_corner_points[4], document_corner_points[5]),
                                      (document_corner_points[6], document_corner_points[7])]

            matrix = self.get_matrix_for_pdf_coordinate_transform(document_corner_points, to_pdf=True)

            lines_objects = []
            for line in lines:

                # Filter out all points that are outside of the document border
                # TODO: Check if performance is better when just removing all points a the end that are negative.
                #  Those are outside too.
                points_on_document = []
                for point in line:
                    if self.is_point_on_document(point[0], point[1], document_corner_tuples):
                        points_on_document.append(point)

                if len(points_on_document) > 0:
                    points_to_be_transformed = np.array([points_on_document], dtype=np.float32)

                    transformed_points = cv2.perspectiveTransform(points_to_be_transformed, matrix)
                    transformed_points = transformed_points.tolist()[0]
                    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
                    points_list = [item for sublist in transformed_points for item in sublist]
                    lines_objects.append(points_list)

            # Add transformed lines to pdf
            # print(lines_objects)
            if len(lines_objects) > 0:
                self.pdf_annotations_service.add_lines_to_pdf(lines_objects)
                self.pdf_annotations_service.write_changes_to_file()

    def locate_document(self, color_image_table, aruco_markers):
        document_corner_points = self.document_locator_service.locate_document(color_image_table, aruco_markers)
        document_found = len(document_corner_points) > 0
        # print(document_corner_points)

        if len(document_corner_points) % 4 != 0:
            print('NOT 8!')

        return document_found, document_corner_points


if __name__ == '__main__':
    debugger = AnalogueDigitalDocumentsDemo()
