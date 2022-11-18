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

MAX_TIME_DOCUMENT_MISSING_SEC = 1

RES_2160P = True

DEBUG_MODE = False


class AnalogueDigitalDocumentsDemo:

    stored_lines = []
    # document_corner_points = []
    converted_document_corner_points = []  # Store current pos of document on table

    lines_on_pdf = []
    lines_on_pdf_modified = False


    highlight_dict = {}
    document_last_seen_timestamp = 0


    def __init__(self):
        self.document_locator_service = DocumentLocatorService()
        self.pdf_annotations_service = PDFAnnotationsService()
        self.fiducials_detection_service = FiducialsDetectorService()

    def get_highlight_rectangles(self, frame, transform_matrix):
        self.transform_matrix = transform_matrix

        highlights, notes, freehand_lines, document_changed = self.pdf_annotations_service.get_annotations()

        # Detect ArUco markers in the camera frame
        aruco_markers = self.fiducials_detection_service.detect_fiducials(frame)

        # Locate the document. Order of document_corner_points TLC, BLC, BRC, TRC
        document_found, document_corner_points = self.locate_document(frame, aruco_markers)

        # self.document_corner_points = document_corner_points

        # Convert the four corner points of the document into the projection space coordinate system
        corner_points_tuple_list = self.list_to_points_list(document_corner_points)
        self.converted_document_corner_points = self.transform_coords_to_output_res(corner_points_tuple_list)

        document_removed = False

        now = time.time()

        if document_found:
            self.document_last_seen_timestamp = time.time()
            all_highlight_points, highlight_ids = self.convert_hightlight_data(highlights)

            self.highlight_dict = self.pdf_points_to_real_world(frame, all_highlight_points, highlight_ids, document_corner_points)
        else:
            # Check how long the document is missing. Remove all projected highlights if missing for too long
            if now - self.document_last_seen_timestamp > MAX_TIME_DOCUMENT_MISSING_SEC:
                document_removed = True
                # TODO: ONLY SEND once
                self.highlight_dict = {}

        if self.lines_on_pdf_modified:
            print('---------------------------------------------LINES ON PDF MODIFIED!')
            # TODO: transform lines on pdf here and send them again
            self.lines_on_pdf_modified = False

        return self.highlight_dict, document_changed, document_removed

    def __check_highlights_removed(self, highlight_ids):
        # Iterate over the list of known highlight IDs. If one does not appear in the list of new highlight IDs, we
        # know that the corresponding highlight must have been deleted
        for previous_highlight_id in self.highlight_dict.keys():
            if previous_highlight_id not in highlight_ids:
                print('Deleted Highlight with ID:', previous_highlight_id)
                return True
        return False

    # def on_new_brio_frame(self, frame, homography_matrix):
    #     self.get_highlight_rectangles(frame, homography_matrix)

    # Convert points from the coordinate system of the pdf to the coordinate system of the output projection
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
        # TODO: THIS HORRIBLE WORKAROUND CURRNENTLY ONLY WORKS FOR NOVEMBER -> 11 FIX THIS ASAP
        return int(timestamp.replace('D:202211', '').replace("+01'00", '') + str(additional_number))

    def transform_coords_to_output_res(self, corner_points_tuple_list):

        for i, point in enumerate(corner_points_tuple_list):
            coords = np.array([point[0], point[1], 1])

            transformed_coords = self.transform_matrix.dot(coords)

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

    def list_to_points_list(self, list_of_x_y_coords):
        points_list = []

        points_x = list_of_x_y_coords[::2]
        points_y = list_of_x_y_coords[1::2]

        for i in range(len(points_x)):
            point = (int(points_x[i]), int(points_y[i]))
            points_list.append(point)

        return points_list

    def on_new_finished_lines(self, lines):

        line_ids_to_delete = []
        remaining_line_points = {}

        if len(lines) > len(self.stored_lines):
            num_new_lines = len(lines) - len(self.stored_lines)
            self.stored_lines = lines.copy()
            # print('Received {} new line(s) to check'. format(num_new_lines))
            new_lines = self.stored_lines[len(self.stored_lines) - num_new_lines:]
            # print('Extracted new lines:', new_lines)

            line_ids_to_delete, remaining_line_points = self.process_new_lines(new_lines)
        return line_ids_to_delete, remaining_line_points

    def process_new_lines(self, new_lines):
        lines_to_add_to_pdf = []
        remaining_line_points = {}
        line_ids_to_delete = []

        for new_line in new_lines:
            for line_id, line in new_line.items():
                print('CHECKING LINE with ID {} and length {}', line_id, len(line))
                points_on_document = []
                points_outside_document = []
                for point in line:
                    if self.__is_point_on_document(point[0], point[1]):
                        points_on_document.append(point)
                    else:
                        points_outside_document.append(point)
                if len(points_on_document) > 0:
                    lines_to_add_to_pdf.append(points_on_document)
                    line_ids_to_delete.append(line_id)
                    if len(points_outside_document) > 0:
                        remaining_line_points[line_id] = points_outside_document

        for line in lines_to_add_to_pdf:
            print('len line points on pdf', len(line))

        for line_id, line in remaining_line_points.items():
            print('len line points off pdf for id {}: '.format(line_id), len(line))

        if len(lines_to_add_to_pdf) > 0:
            print('ADD THESE LINES TO PDF:', lines_to_add_to_pdf)

            self.add_lines_to_pdf(lines_to_add_to_pdf)

        return line_ids_to_delete, remaining_line_points

    # Check if a point is on which side of a line. Return True if on right side, return False if on left side
    # https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
    def __is_on_right_side(self, x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a * x0 - b * y0
        return a * x + b * y + c >= 0

    # This function will check if a point is inside the borders of the document
    # https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
    def __is_point_on_document(self, x, y):
        # self.converted_document_corner_points is already in the projector coordinate space.
        # So the same space as the received lines
        if len(self.converted_document_corner_points) == 0:
            print('No document on table. So line cant be on the document')
            return False
        num_vert = len(self.converted_document_corner_points)
        is_right = [self.__is_on_right_side(x, y, self.converted_document_corner_points[i], self.converted_document_corner_points[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right

    def add_lines_to_pdf(self, lines_to_add_to_pdf):

        if len(lines_to_add_to_pdf) > 0 and len(self.converted_document_corner_points) > 0:
            converted_document_corner_points_flat_list = [item for sublist in self.converted_document_corner_points for item in sublist]
            matrix = self.get_matrix_for_pdf_coordinate_transform(converted_document_corner_points_flat_list, to_pdf=True)

            for line in lines_to_add_to_pdf:
                if len(line) > 0:
                    points_to_be_transformed = np.array([line], dtype=np.float32)

                    transformed_points = cv2.perspectiveTransform(points_to_be_transformed, matrix)
                    transformed_points = transformed_points.tolist()[0]
                    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
                    points_list = [item for sublist in transformed_points for item in sublist]
                    self.lines_on_pdf.append(points_list)
                    self.lines_on_pdf_modified = True

            # Add lines permanently to pdf file
            if len(self.lines_on_pdf) > 0:
                print('Writing {} lines to pdf'.format(len(self.lines_on_pdf)))
                self.pdf_annotations_service.add_lines_to_pdf(self.lines_on_pdf)
                self.pdf_annotations_service.write_changes_to_file()

    def get_matrix_for_pdf_coordinate_transform(self, document_corner_points, to_pdf):

        # document_corner_points needs to be a flat list!
        # print('Document Corners in matrix:', document_corner_points)

        document_corners = np.float32([[document_corner_points[0], document_corner_points[1]],
                                       [document_corner_points[2], document_corner_points[3]],
                                       [document_corner_points[4], document_corner_points[5]],
                                       [document_corner_points[6], document_corner_points[7]]])

        if to_pdf:
            # Use different order here to have the correct orientation of the points. No idea why...
            pdf_res = np.float32([[0, PDF_HEIGHT], [0, 0], [PDF_WIDTH, 0], [PDF_WIDTH, PDF_HEIGHT]])
            matrix = cv2.getPerspectiveTransform(document_corners, pdf_res)
        else:
            pdf_res = np.float32([[0, 0], [0, PDF_HEIGHT], [PDF_WIDTH, PDF_HEIGHT], [PDF_WIDTH, 0]])
            matrix = cv2.getPerspectiveTransform(pdf_res, document_corners)

        # This matrix will be used to transform the points into the correct coordinate space for the pdf
        # (with origin in the bottom left corner and the correct resolution of the pdf)

        return matrix

    def locate_document(self, color_image_table, aruco_markers):
        document_corner_points = self.document_locator_service.locate_document(color_image_table, aruco_markers)
        document_found = len(document_corner_points) > 0
        return document_found, document_corner_points


if __name__ == '__main__':
    debugger = AnalogueDigitalDocumentsDemo()
