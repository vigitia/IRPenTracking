# Author: Jürgen Hahn

import math

import cv2
import numpy as np

from surface_extractor import SurfaceExtractor

class PenColorDetector:
    start = True





    def update_pen_thresh(thresh):
        PenColorDetector.pen_tresh = thresh

    def update_red_lower_r(rval):
        PenColorDetector.red_lower[0] = rval

    def update_red_lower_g(gval):
        PenColorDetector.red_lower[1] = gval

    def update_red_lower_b(bval):
        PenColorDetector.red_lower[2] = bval

    def update_red_upper_r(rval):
        PenColorDetector.red_upper[0] = rval

    def update_red_upper_g(gval):
        PenColorDetector.red_upper[1] = gval

    def update_red_upper_b(bval):
        PenColorDetector.red_upper[2] = bval



    def update_green_lower_r(rval):
        PenColorDetector.green_lower[0] = rval

    def update_green_lower_g(gval):
        PenColorDetector.green_lower[1] = gval

    def update_green_lower_b(bval):
        PenColorDetector.green_lower[2] = bval

    def update_green_upper_r(rval):
        PenColorDetector.green_upper[0] = rval

    def update_green_upper_g(gval):
        PenColorDetector.green_upper[1] = gval

    def update_green_upper_b(bval):
        PenColorDetector.green_upper[2] = bval



    def update_blue_lower_r(rval):
        PenColorDetector.blue_lower[0] = rval

    def update_blue_lower_g(gval):
        PenColorDetector.blue_lower[1] = gval

    def update_blue_lower_b(bval):
        PenColorDetector.blue_lower[2] = bval

    def update_blue_upper_r(rval):
        PenColorDetector.blue_upper[0] = rval

    def update_blue_upper_g(gval):
        PenColorDetector.blue_upper[1] = gval

    def update_blue_upper_b(bval):
        PenColorDetector.blue_upper[2] = bval



    def update_black_lower_r(rval):
        PenColorDetector.black_lower[0] = rval

    def update_black_lower_g(gval):
        PenColorDetector.black_lower[1] = gval

    def update_black_lower_b(bval):
        PenColorDetector.black_lower[2] = bval

    def update_black_upper_r(rval):
        PenColorDetector.black_upper[0] = rval

    def update_black_upper_g(gval):
        PenColorDetector.black_upper[1] = gval

    def update_black_upper_b(bval):
        PenColorDetector.black_upper[2] = bval

    red_lower = np.array([112, 84, 83])
    red_upper = np.array([183, 195, 221])

    green_lower = np.array([55, 42, 55])
    green_upper = np.array([83, 176, 222])

    blue_lower = np.array([88, 62, 66])
    blue_upper = np.array([123, 217, 255])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([255, 255, 255])

    pen_tresh = 0

    @staticmethod
    def detect(frame, ids_and_points, DEBUG=False):
        def setup_sliders():
            #cv2.namedWindow('sliders')
            # cv2.createTrackbar('Pen Thresh', 'sliders', 0, 255, lambda x: PenColorDetector.update_pen_thresh(x))

            cv2.createTrackbar('Red Lower H', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_lower_r(x))
            cv2.setTrackbarPos('Red Lower H', 'sliders', PenColorDetector.red_lower[0])
            cv2.createTrackbar('Red Lower S', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_lower_g(x))
            cv2.setTrackbarPos('Red Lower S', 'sliders', PenColorDetector.red_lower[1])
            cv2.createTrackbar('Red Lower V', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_lower_b(x))
            cv2.setTrackbarPos('Red Lower V', 'sliders', PenColorDetector.red_lower[2])
            cv2.createTrackbar('Red Upper H', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_upper_r(x))
            cv2.setTrackbarPos('Red Upper H', 'sliders', PenColorDetector.red_upper[0])
            cv2.createTrackbar('Red Upper S', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_upper_g(x))
            cv2.setTrackbarPos('Red Upper S', 'sliders', PenColorDetector.red_upper[1])
            cv2.createTrackbar('Red Upper V', 'sliders', 0, 255, lambda x: PenColorDetector.update_red_upper_b(x))
            cv2.setTrackbarPos('Red Upper V', 'sliders', PenColorDetector.red_upper[2])

            cv2.createTrackbar('Green Lower H', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_lower_r(x))
            cv2.setTrackbarPos('Green Lower H', 'sliders', PenColorDetector.green_lower[0])
            cv2.createTrackbar('Green Lower S', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_lower_g(x))
            cv2.setTrackbarPos('Green Lower S', 'sliders', PenColorDetector.green_lower[1])
            cv2.createTrackbar('Green Lower V', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_lower_b(x))
            cv2.setTrackbarPos('Green Lower V', 'sliders', PenColorDetector.green_lower[2])
            cv2.createTrackbar('Green Upper H', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_upper_r(x))
            cv2.setTrackbarPos('Green Upper H', 'sliders', PenColorDetector.green_upper[0])
            cv2.createTrackbar('Green Upper S', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_upper_g(x))
            cv2.setTrackbarPos('Green Upper S', 'sliders', PenColorDetector.green_upper[1])
            cv2.createTrackbar('Green Upper V', 'sliders', 0, 255, lambda x: PenColorDetector.update_green_upper_b(x))
            cv2.setTrackbarPos('Green Upper V', 'sliders', PenColorDetector.green_upper[2])

            cv2.createTrackbar('Blue Lower H', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_lower_r(x))
            cv2.setTrackbarPos('Blue Lower H', 'sliders', PenColorDetector.blue_lower[0])
            cv2.createTrackbar('Blue Lower S', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_lower_g(x))
            cv2.setTrackbarPos('Blue Lower S', 'sliders', PenColorDetector.blue_lower[1])
            cv2.createTrackbar('Blue Lower V', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_lower_b(x))
            cv2.setTrackbarPos('Blue Lower V', 'sliders', PenColorDetector.blue_lower[2])
            cv2.createTrackbar('Blue Upper H', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_upper_r(x))
            cv2.setTrackbarPos('Blue Upper H', 'sliders', PenColorDetector.blue_upper[0])
            cv2.createTrackbar('Blue Upper S', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_upper_g(x))
            cv2.setTrackbarPos('Blue Upper S', 'sliders', PenColorDetector.blue_upper[1])
            cv2.createTrackbar('Blue Upper V', 'sliders', 0, 255, lambda x: PenColorDetector.update_blue_upper_b(x))
            cv2.setTrackbarPos('Blue Upper V', 'sliders', PenColorDetector.blue_upper[2])

            cv2.createTrackbar('Black Lower H', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_lower_r(x))
            cv2.setTrackbarPos('Black Lower H', 'sliders', PenColorDetector.black_lower[0])
            cv2.createTrackbar('Black Lower S', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_lower_g(x))
            cv2.setTrackbarPos('Black Lower S', 'sliders', PenColorDetector.black_lower[1])
            cv2.createTrackbar('Black Lower V', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_lower_b(x))
            cv2.setTrackbarPos('Black Lower V', 'sliders', PenColorDetector.black_lower[2])
            cv2.createTrackbar('Black Upper H', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_upper_r(x))
            cv2.setTrackbarPos('Black Upper H', 'sliders', PenColorDetector.black_upper[0])
            cv2.createTrackbar('Black Upper S', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_upper_g(x))
            cv2.setTrackbarPos('Black Upper S', 'sliders', PenColorDetector.black_upper[1])
            cv2.createTrackbar('Black Upper V', 'sliders', 0, 255, lambda x: PenColorDetector.update_black_upper_b(x))
            cv2.setTrackbarPos('Black Upper V', 'sliders', PenColorDetector.black_upper[2])


        def accumulate_contours(contours):
            dest = []
            if contours:
                dest += contours

            return dest

        def associate_pens_tips(tips, r, g, b):
            pens = {0: 9999, 1: 9999, 2: 9999}

            for k, pen_contour in enumerate([r, g, b]):
                distance = math.inf
                idx = 9999
                cidx = 9999
                for i, wc in enumerate(tips):
                    rect = cv2.minAreaRect(wc)
                    wbox = cv2.boxPoints(rect)
                    wbox = np.int0(wbox)

                    if pen_contour is not None and len(pen_contour) > 0:
                        rect = cv2.minAreaRect(pen_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        distances = []
                        for p1 in wbox:
                            for p2 in box:
                                dist = np.linalg.norm(p1 - p2)
                                distances.append(dist)

                        if min(distances) < distance and min(distances) < 30:
                            distance = min(distances)
                            idx = i
                            cidx = k

                pens[cidx] = idx

            if 9999 in pens:
                del pens[9999]
            return pens[0], pens[1], pens[2]


        def get_pen_tip_association_line(pen_contour, tip_contour):
            rect = cv2.minAreaRect(tip_contour)
            wbox = cv2.boxPoints(rect)
            wbox = np.int0(wbox)

            rect = cv2.minAreaRect(pen_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            pen_box_center = tuple(np.mean(box, axis=0).astype(int))
            associated_white_box_center = tuple(np.mean(wbox, axis=0).astype(int))

            return pen_box_center, associated_white_box_center


        def draw_pen_tip_association(pen_contour, tip_contour):
            pen_box_center, associated_white_box_center = get_pen_tip_association_line(pen_contour, tip_contour)
            cv2.line(frame, pen_box_center, associated_white_box_center, (0, 255, 255), 2)


        def draw_pen_tip_associations(r, g, b, tips, wc_i_r, wc_i_g, wc_i_b):
            if r is not None and len(r) > 0 and tips and len(tips) > wc_i_r and wc_i_r != 9999:
                draw_pen_tip_association(r, tips[wc_i_r])

            if g is not None and len(g) > 0 and tips and len(tips) > wc_i_g and wc_i_g != 9999:
                draw_pen_tip_association(g, tips[wc_i_g])

            if b is not None and len(b) > 0 and tips and len(tips) > wc_i_b and wc_i_b != 9999:
                draw_pen_tip_association(b, tips[wc_i_b])

        def determine_approx_lines_of_pens(r, g, b, tips_contours, wc_i_r, wc_i_g, wc_i_b):
            pens = {}
            if r is not None and len(r) > 0 and tips_contours and len(tips_contours) > wc_i_r and wc_i_r != 9999:
                red_pen_box_center, red_pen_tip_center = get_pen_tip_association_line(r, tips_contours[wc_i_r])
                pens["r"] = {"pen": red_pen_box_center, "tip": red_pen_tip_center}

            if g is not None and len(g) > 0 and tips_contours and len(tips_contours) > wc_i_g and wc_i_g != 9999:
                green_pen_box_center, green_pen_tip_center = get_pen_tip_association_line(g, tips_contours[wc_i_g])
                pens["g"] = {"pen": green_pen_box_center, "tip": green_pen_tip_center}

            if b is not None and len(b) > 0 and tips_contours and len(tips_contours) > wc_i_b and wc_i_b != 9999:
                blue_pen_box_center, blue_pen_tip_center = get_pen_tip_association_line(b, tips_contours[wc_i_b])
                pens["b"] = {"pen": blue_pen_box_center, "tip": blue_pen_tip_center}

            return pens

        def distance_point_line(p, l):
            x0, y0 = p
            x1, y1, = l[0]
            x2, y2, = l[1]

            return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if PenColorDetector.start:
            PenColorDetector.start = False
            setup_sliders()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r = cv2.inRange(hsv, PenColorDetector.red_lower, PenColorDetector.red_upper)
        g = cv2.inRange(hsv, PenColorDetector.green_lower, PenColorDetector.green_upper)
        b = cv2.inRange(hsv, PenColorDetector.blue_lower, PenColorDetector.blue_upper)

        r = cv2.erode(r, np.ones((5, 5), np.uint8))
        r = cv2.dilate(r, np.ones((5, 5), np.uint8), iterations=3)
        g = cv2.erode(g, np.ones((5, 5), np.uint8))
        g = cv2.dilate(g, np.ones((5, 5), np.uint8), iterations=3)
        b = cv2.erode(b, np.ones((5, 5), np.uint8))
        b = cv2.dilate(b, np.ones((5, 5), np.uint8), iterations=3)

        rcontours, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gcontours, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bcontours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, s, v = cv2.split(hsv)

        thresh_value = 65
        max_value = 255
        _, binary_img = cv2.threshold(v, thresh_value, max_value, cv2.THRESH_BINARY_INV)
        binary_img = cv2.erode(binary_img, np.ones((5, 5), np.uint8))
        binary_img = cv2.dilate(binary_img, np.ones((5, 5), np.uint8), iterations=3)

        black_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_contours = accumulate_contours(rcontours)
        green_contours = accumulate_contours(gcontours)
        blue_contours = accumulate_contours(bcontours)

        cv2.drawContours(frame, black_contours, -1, (0, 0, 0), 3)

        if len(red_contours) > 0:
            red_contours = np.vstack(red_contours)
            rect = cv2.minAreaRect(red_contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(frame, [box], -1, (0, 0, 255), 3)

        if len(green_contours) > 0:
            green_contours = np.vstack(green_contours)
            rect = cv2.minAreaRect(green_contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)

        if len(blue_contours) > 0:
            blue_contours = np.vstack(blue_contours)
            rect = cv2.minAreaRect(blue_contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(frame, [box], -1, (255, 0, 0), 3)

        wc_i_r, wc_i_g, wc_i_b = associate_pens_tips(black_contours, red_contours, green_contours, blue_contours)

        pens = determine_approx_lines_of_pens(red_contours, green_contours, blue_contours, black_contours, wc_i_r, wc_i_g, wc_i_b)
        draw_pen_tip_associations(red_contours, green_contours, blue_contours, black_contours, wc_i_r, wc_i_g, wc_i_b)


        id_pen_association = {}

        for _id, x, y in ids_and_points:
            px = x / 3840
            py = y / 2160

            px *= 1280
            py *= 720

            p = px, py

            id_pen_association[_id] = {"color": None}
            color = None
            distance = math.inf

            for k, v in pens.items():
                l = [v["pen"], v["tip"]]

                d = distance_point_line(p, l)

                if d < distance:
                    color = k
                    distance = d

            id_pen_association[_id]["color"] = color

        return id_pen_association


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        cv2.getTrackbarPos('Red Lower R', 'sliders')

        # Capture frame-by-frame
        ret, frame = cap.read()


        # pens = PenColorDetector.detect(frame, True)
        pens = PenColorDetector.detect(frame, True)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
