import cv2
import numpy as np

from DocumentLocatorService import DocumentLocatorService
from PDFAnnotationsService import PDFAnnotationsService
from FiducialsDetectorService import FiducialsDetectorService
from logitech_brio import LogitechBrio

from scipy.spatial import distance

PDF_WIDTH = 595.446
PDF_HEIGHT = 841.691


class AnalogueDigitalDocumentsDemo:

    def __init__(self):

        self.document_locator_service = DocumentLocatorService()
        self.pdf_annotations_service = PDFAnnotationsService()
        self.fiducials_detection_service = FiducialsDetectorService()

        self.logitech_brio_camera = LogitechBrio(self)
        self.logitech_brio_camera.init_video_capture()
        self.logitech_brio_camera.start()

    def on_new_brio_frame(self, frame, homography_matrix):
        highlights, notes, freehand_lines = self.pdf_annotations_service.get_annotations()

        # Detect ArUco markers
        aruco_markers = self.fiducials_detection_service.detect_fiducials(frame)

        # Locate the document. Order of document_corner_points TLC, BLC, BRC, TRC
        document_found, document_corner_points = self.locate_document(frame, aruco_markers)

        if document_found:
            frame = self.pdf_points_to_real_world(frame, highlights, document_corner_points)

            # frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]), (document_corner_points[2], document_corner_points[3]), (0, 0, 0), thickness=3)
            # frame = cv2.line(frame, (document_corner_points[2], document_corner_points[3]), (document_corner_points[4], document_corner_points[5]), (0, 0, 0), thickness=3)
            # frame = cv2.line(frame, (document_corner_points[4], document_corner_points[5]), (document_corner_points[6], document_corner_points[7]), (0, 0, 0), thickness=3)
            # frame = cv2.line(frame, (document_corner_points[0], document_corner_points[1]), (document_corner_points[6], document_corner_points[7]), (0, 0, 0), thickness=3)

        cv2.imshow('Logitech Brio', frame)
        cv2.waitKey(1)

    def pdf_points_to_real_world(self, frame, highlights, document_corner_points):

        matrix = self.get_matrix_for_pdf_coordinate_transform(document_corner_points, to_pdf=False)

        all_highlight_points = []

        for highlight in highlights:
            highlight_points = self.list_to_points_list(highlight['quad_points'][0])
            # for highlight_point in highlight_points:
            #     cv2.circle(frame, highlight_point, 5, (0, 0, 0), -1)

            all_highlight_points.append(highlight_points)

        print(all_highlight_points)

        for highlight_group in all_highlight_points:
            points_to_be_transformed = np.array([highlight_group], dtype=np.float32)

            transformed_points = cv2.perspectiveTransform(points_to_be_transformed, matrix)
            transformed_points = transformed_points.tolist()[0]
            # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
            flat_list = [item for sublist in transformed_points for item in sublist]

            points_tuple_list = self.list_to_points_list(flat_list)
            cv2.fillPoly(frame, pts=[np.array(points_tuple_list)], color=(0, 255, 255))
            # cv2.rectangle(frame, points_tuple_list[0], points_tuple_list[2], (0, 255, 255), -1)

            # for point in points_tuple_list:
            #     cv2.circle(frame, point, 5, (0, 0, 0), -1)

            print('points_list after transform', points_tuple_list)

        return frame

    def list_to_points_list(self, list_of_x_y_coords):
        points_list = []

        points_x = list_of_x_y_coords[::2]
        points_y = list_of_x_y_coords[1::2]

        for i in range(len(points_x)):
            point = (int(points_x[i]), int(points_y[i]))
            points_list.append(point)

        return points_list


    def transform_pdf_point_to_real_world_document(self, pdf_x, pdf_y, document_corner_points):

        # Calculate distance between top left corner and top right corner
        doc_width = distance.euclidean((document_corner_points[0], document_corner_points[1]),
                                       (document_corner_points[6], document_corner_points[7]))

        # Calculate distance between top left corner and bottom left corner
        doc_height = distance.euclidean((document_corner_points[0], document_corner_points[1]),
                                        (document_corner_points[2], document_corner_points[3]))

        width_fraction = doc_width / PDF_WIDTH
        height_fraction = doc_height / PDF_HEIGHT

        x_new = pdf_x * width_fraction
        y_new = pdf_y * height_fraction

        x_axis_normalized = None
        y_axis_normalized = None



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

        # This matrix will be used to transform the points into the correct coordinate space for the pdf
        # (with origin in the bottom left corner and the correct resolution of the pdf)

        if to_pdf:
            matrix = cv2.getPerspectiveTransform(document_corners, pdf_res)
        else:
            matrix = cv2.getPerspectiveTransform(pdf_res, document_corners)

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
            print(lines_objects)
            if len(lines_objects) > 0:
                self.pdf_annotations_service.add_lines_to_pdf(lines_objects)
                self.pdf_annotations_service.write_changes_to_file()

    def locate_document(self, color_image_table, aruco_markers):
        document_corner_points = self.document_locator_service.locate_document(color_image_table, aruco_markers)
        document_found = len(document_corner_points) > 0
        print(document_corner_points)

        if len(document_corner_points) % 4 != 0:
            print('NOT 8!')

        return document_found, document_corner_points


if __name__ == '__main__':
    debugger = AnalogueDigitalDocumentsDemo()
