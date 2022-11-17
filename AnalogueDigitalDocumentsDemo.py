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

    stored_lines = []
    converted_document_corner_points = []  # Store current pos of document on table

    highlight_dict = {}

    def __init__(self):
        self.document_locator_service = DocumentLocatorService()
        self.pdf_annotations_service = PDFAnnotationsService()
        self.fiducials_detection_service = FiducialsDetectorService()

    def get_highlight_rectangles(self, frame, transform_matrix):
        self.transform_matrix = transform_matrix

        highlights, notes, freehand_lines = self.pdf_annotations_service.get_annotations()

        # Detect ArUco markers in the camera frame
        aruco_markers = self.fiducials_detection_service.detect_fiducials(frame)

        # Locate the document. Order of document_corner_points TLC, BLC, BRC, TRC
        document_found, document_corner_points = self.locate_document(frame, aruco_markers)

        # Convert the four corner points of the document into the projection space coordinate system
        corner_points_tuple_list = self.list_to_points_list(document_corner_points)
        self.converted_document_corner_points = self.transform_coords_to_output_res(corner_points_tuple_list)

        highlights_removed = False
        # TODO: CHECK DOCUMENT MISSING FOR SOME TIME

        if document_found:
            all_highlight_points, highlight_ids = self.convert_hightlight_data(highlights)
            # removed_highlight_ids = self.get_removed_highlight_ids(highlights)
            highlights_removed = self.check_highlights_removed(highlight_ids)

            self.highlight_dict = self.pdf_points_to_real_world(frame, all_highlight_points, highlight_ids, document_corner_points)

        return self.highlight_dict, highlights_removed

    def check_highlights_removed(self, highlight_ids):
        # Iterate over the list of known highlight IDs. If one does not appear in the list of new highlight IDs, we
        # know that the corresponding highlight must have been deleted
        for previous_highlight_id in self.highlight_dict.keys():
            if previous_highlight_id not in highlight_ids:
                print('Deleted Highlight with ID:', previous_highlight_id)
                return True
        return False


    def get_removed_highlight_ids(self, new_highlights):
        removed_highlight_ids = []
        # for new_highlight in new_highlights:
        #     new_highlight_id = self.
        #     if new_highlight[]

        return removed_highlight_ids


    def on_new_brio_frame(self, frame, homography_matrix):
        self.get_highlight_rectangles(frame, homography_matrix)

    def pdf_points_to_real_world(self, frame, all_highlight_points, highlight_ids, document_corner_points):

        matrix = self.get_matrix_for_pdf_coordinate_transform(document_corner_points, to_pdf=False)

        highlight_dict = {}

        for i, highlight_group in enumerate(all_highlight_points):
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
            flat_list = [int(number) for number in flat_list]

            highlight_dict[highlight_ids[i]] = flat_list

            if DEBUG_MODE:
                cv2.fillPoly(frame, pts=[np.array(points_tuple_list)], color=(0, 255, 255))

        if DEBUG_MODE:
            frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]),
                             (document_corner_points[2], document_corner_points[3]), (0, 0, 255), thickness=3)
            frame = cv2.line(frame, (document_corner_points[2], document_corner_points[3]),
                             (document_corner_points[4], document_corner_points[5]), (0, 0, 255), thickness=3)
            frame = cv2.line(frame, (document_corner_points[4], document_corner_points[5]),
                             (document_corner_points[6], document_corner_points[7]), (0, 0, 255), thickness=3)
            frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]),
                             (document_corner_points[6], document_corner_points[7]), (0, 0, 255), thickness=3)

            cv2.imshow('Logitech Brio', frame)
            cv2.waitKey(1)

        return highlight_dict



    def convert_hightlight_data(self, highlights):
        highlight_ids = []
        all_highlight_points = []

        for highlight_object in highlights:
            highlight_group = highlight_object['quad_points']
            for i, highlight_list in enumerate(highlight_group):
                highlight_ids.append(self.generate_id_from_timestamp(highlight_object['timestamp'], i))
                highlight_points = self.list_to_points_list(highlight_list)
                all_highlight_points.append(highlight_points)

        return all_highlight_points, highlight_ids

    # Create a temporary ID for the highlight from the timestamp
    # TODO: Improve this!
    def generate_id_from_timestamp(self, timestamp, additional_number):
        return int(timestamp.replace('D:202211', '').replace("+01'00", '') + str(additional_number))

    def transform_coords_to_output_res(self, corner_points_tuple_list):

        #try:
        for i, point in enumerate(corner_points_tuple_list):
            coords = np.array([point[0], point[1], 1])

            transformed_coords = self.transform_matrix.dot(coords)

            RES_2160P = True

            if RES_2160P:
                # Normalize coordinates by dividing by z
                # TODO: Improve this conversion. currently it only scales up from 1080p to 2160p by multiplying by 2
                corner_points_tuple_list[i] = ((transformed_coords[0] / transformed_coords[2]) * 2,
                                        (transformed_coords[1] / transformed_coords[2]) * 2)
            else:
                # Normalize coordinates by dividing by z
                corner_points_tuple_list[i] = ((transformed_coords[0] / transformed_coords[2]),
                                        (transformed_coords[1] / transformed_coords[2]))

        return corner_points_tuple_list
        # except Exception as e:
        #     print(e)
        #     print('Current transform matrix:', self.transform_matrix)
        #     print('Error in transform_coords_to_output_res(). Maybe the transform_matrix is malformed?')
        #     print('This error could also appear if CALIBRATION_MODE is still enabled in logitech.py')
        #     time.sleep(5)
        #     sys.exit(1)

    def list_to_points_list(self, list_of_x_y_coords):
        points_list = []

        points_x = list_of_x_y_coords[::2]
        points_y = list_of_x_y_coords[1::2]

        for i in range(len(points_x)):
            point = (int(points_x[i]), int(points_y[i]))
            points_list.append(point)

        return points_list

    def on_new_finished_line(self, lines):
        if len(lines) > len(self.stored_lines):
            num_new_lines = len(lines) - len(self.stored_lines)
            self.stored_lines = lines.copy()
            print('Received {} new line(s) to check'. format(num_new_lines))
            new_lines = self.stored_lines[len(self.stored_lines) - num_new_lines:]
            print('Extracted new lines:', new_lines)

    def process_new_lines(self, new_lines):
        for line in new_lines:
            print('Line', line)
            for point in line:
                print('Point', point)

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
    def is_point_on_document(self, x, y):
        num_vert = len(self.converted_document_corner_points)
        is_right = [self.is_on_right_side(x, y, self.converted_document_corner_points[i], self.converted_document_corner_points[(i + 1) % num_vert]) for i in range(num_vert)]
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
