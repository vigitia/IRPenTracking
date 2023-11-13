# Author: Jürgen Hahn
import math

import cv2
import numpy as np

DEBUG = True


class PenColorDetector:
    def __init__(self):
        # Load profile SI_PENS_2 in GUVCVIEW
        # PING PONG BALLS
        self.red_lower = np.array([162, 96, 109])
        self.red_upper = np.array([183, 176, 223])
        self.green_lower = np.array([68, 57, 111])
        self.green_upper = np.array([84, 164, 157])
        self.blue_lower = np.array([101, 140, 124])
        self.blue_upper = np.array([110, 198, 215])
        self.white_lower = np.array([101, 140, 124])
        self.white_upper = np.array([110, 198, 215])
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([210, 110, 79])

        # self.red_lower = np.array([0, 121, 140])
        # self.red_upper = np.array([11, 183, 193])
        # self.green_lower = np.array([37, 20, 114])
        # self.green_upper = np.array([79, 131, 157])
        # self.blue_lower = np.array([100, 53, 152])
        # self.blue_upper = np.array([106, 179, 221])

        # self.red_lower = np.array([146, 45, 60])
        # self.red_upper = np.array([184, 139, 112])
        # self.green_lower = np.array([65, 108, 60])
        # self.green_upper = np.array([93, 172, 106])
        # self.blue_lower = np.array([87, 176, 98])
        # self.blue_upper = np.array([105, 218, 221])

        self.start = True

    def update_red_lower_r(self, rval):
        self.red_lower[0] = rval

    def update_red_lower_g(self, gval):
        self.red_lower[1] = gval

    def update_red_lower_b(self, bval):
        self.red_lower[2] = bval

    def update_red_upper_r(self, rval):
        self.red_upper[0] = rval

    def update_red_upper_g(self, gval):
        self.red_upper[1] = gval

    def update_red_upper_b(self, bval):
        self.red_upper[2] = bval



    def update_green_lower_r(self, rval):
        self.green_lower[0] = rval

    def update_green_lower_g(self, gval):
        self.green_lower[1] = gval

    def update_green_lower_b(self, bval):
        self.green_lower[2] = bval

    def update_green_upper_r(self, rval):
        self.green_upper[0] = rval

    def update_green_upper_g(self, gval):
        self.green_upper[1] = gval

    def update_green_upper_b(self, bval):
        self.green_upper[2] = bval



    def update_blue_lower_r(self, rval):
        self.blue_lower[0] = rval

    def update_blue_lower_g(self, gval):
        self.blue_lower[1] = gval

    def update_blue_lower_b(self, bval):
        self.blue_lower[2] = bval

    def update_blue_upper_r(self, rval):
        self.blue_upper[0] = rval

    def update_blue_upper_g(self, gval):
        self.blue_upper[1] = gval

    def update_blue_upper_b(self, bval):
        self.blue_upper[2] = bval


    def update_white_lower_r(self, rval):
        self.white_lower[0] = rval

    def update_white_lower_g(self, gval):
        self.white_lower[1] = gval

    def update_white_lower_b(self, bval):
        self.white_lower[2] = bval

    def update_white_upper_r(self, rval):
        self.white_upper[0] = rval

    def update_white_upper_g(self, gval):
        self.white_upper[1] = gval

    def update_white_upper_b(self, bval):
        self.white_upper[2] = bval



    def update_black_lower_r(self, rval):
        self.black_lower[0] = rval

    def update_black_lower_g(self, gval):
        self.black_lower[1] = gval

    def update_black_lower_b(self, bval):
        self.black_lower[2] = bval

    def update_black_upper_r(self, rval):
        self.black_upper[0] = rval

    def update_black_upper_g(self, gval):
        self.black_upper[1] = gval

    def update_black_upper_b(self, bval):
        self.black_upper[2] = bval

    def setup_sliders(self):
        cv2.namedWindow('sliders')
        # cv2.createTrackbar('Pen Thresh', 'sliders', 0, 255, lambda x: self.update_pen_thresh(x))

        cv2.createTrackbar('Red Lower H', 'sliders', 0, 255, lambda x: self.update_red_lower_r(x))
        cv2.setTrackbarPos('Red Lower H', 'sliders', self.red_lower[0])
        cv2.createTrackbar('Red Lower S', 'sliders', 0, 255, lambda x: self.update_red_lower_g(x))
        cv2.setTrackbarPos('Red Lower S', 'sliders', self.red_lower[1])
        cv2.createTrackbar('Red Lower V', 'sliders', 0, 255, lambda x: self.update_red_lower_b(x))
        cv2.setTrackbarPos('Red Lower V', 'sliders', self.red_lower[2])
        cv2.createTrackbar('Red Upper H', 'sliders', 0, 255, lambda x: self.update_red_upper_r(x))
        cv2.setTrackbarPos('Red Upper H', 'sliders', self.red_upper[0])
        cv2.createTrackbar('Red Upper S', 'sliders', 0, 255, lambda x: self.update_red_upper_g(x))
        cv2.setTrackbarPos('Red Upper S', 'sliders', self.red_upper[1])
        cv2.createTrackbar('Red Upper V', 'sliders', 0, 255, lambda x: self.update_red_upper_b(x))
        cv2.setTrackbarPos('Red Upper V', 'sliders', self.red_upper[2])

        cv2.createTrackbar('Green Lower H', 'sliders', 0, 255, lambda x: self.update_green_lower_r(x))
        cv2.setTrackbarPos('Green Lower H', 'sliders', self.green_lower[0])
        cv2.createTrackbar('Green Lower S', 'sliders', 0, 255, lambda x: self.update_green_lower_g(x))
        cv2.setTrackbarPos('Green Lower S', 'sliders', self.green_lower[1])
        cv2.createTrackbar('Green Lower V', 'sliders', 0, 255, lambda x: self.update_green_lower_b(x))
        cv2.setTrackbarPos('Green Lower V', 'sliders', self.green_lower[2])
        cv2.createTrackbar('Green Upper H', 'sliders', 0, 255, lambda x: self.update_green_upper_r(x))
        cv2.setTrackbarPos('Green Upper H', 'sliders', self.green_upper[0])
        cv2.createTrackbar('Green Upper S', 'sliders', 0, 255, lambda x: self.update_green_upper_g(x))
        cv2.setTrackbarPos('Green Upper S', 'sliders', self.green_upper[1])
        cv2.createTrackbar('Green Upper V', 'sliders', 0, 255, lambda x: self.update_green_upper_b(x))
        cv2.setTrackbarPos('Green Upper V', 'sliders', self.green_upper[2])

        cv2.createTrackbar('Blue Lower H', 'sliders', 0, 255, lambda x: self.update_blue_lower_r(x))
        cv2.setTrackbarPos('Blue Lower H', 'sliders', self.blue_lower[0])
        cv2.createTrackbar('Blue Lower S', 'sliders', 0, 255, lambda x: self.update_blue_lower_g(x))
        cv2.setTrackbarPos('Blue Lower S', 'sliders', self.blue_lower[1])
        cv2.createTrackbar('Blue Lower V', 'sliders', 0, 255, lambda x: self.update_blue_lower_b(x))
        cv2.setTrackbarPos('Blue Lower V', 'sliders', self.blue_lower[2])
        cv2.createTrackbar('Blue Upper H', 'sliders', 0, 255, lambda x: self.update_blue_upper_r(x))
        cv2.setTrackbarPos('Blue Upper H', 'sliders', self.blue_upper[0])
        cv2.createTrackbar('Blue Upper S', 'sliders', 0, 255, lambda x: self.update_blue_upper_g(x))
        cv2.setTrackbarPos('Blue Upper S', 'sliders', self.blue_upper[1])
        cv2.createTrackbar('Blue Upper V', 'sliders', 0, 255, lambda x: self.update_blue_upper_b(x))
        cv2.setTrackbarPos('Blue Upper V', 'sliders', self.blue_upper[2])

        cv2.createTrackbar('White Lower H', 'sliders', 0, 255, lambda x: self.update_white_lower_r(x))
        cv2.setTrackbarPos('White Lower H', 'sliders', self.white_lower[0])
        cv2.createTrackbar('White Lower S', 'sliders', 0, 255, lambda x: self.update_white_lower_g(x))
        cv2.setTrackbarPos('White Lower S', 'sliders', self.white_lower[1])
        cv2.createTrackbar('White Lower V', 'sliders', 0, 255, lambda x: self.update_white_lower_b(x))
        cv2.setTrackbarPos('White Lower V', 'sliders', self.white_lower[2])
        cv2.createTrackbar('White Upper H', 'sliders', 0, 255, lambda x: self.update_white_upper_r(x))
        cv2.setTrackbarPos('White Upper H', 'sliders', self.white_upper[0])
        cv2.createTrackbar('White Upper S', 'sliders', 0, 255, lambda x: self.update_white_upper_g(x))
        cv2.setTrackbarPos('White Upper S', 'sliders', self.white_upper[1])
        cv2.createTrackbar('White Upper V', 'sliders', 0, 255, lambda x: self.update_white_upper_b(x))
        cv2.setTrackbarPos('White Upper V', 'sliders', self.white_upper[2])

        cv2.createTrackbar('Black Lower H', 'sliders', 0, 255, lambda x: self.update_black_lower_r(x))
        cv2.setTrackbarPos('Black Lower H', 'sliders', self.black_lower[0])
        cv2.createTrackbar('Black Lower S', 'sliders', 0, 255, lambda x: self.update_black_lower_g(x))
        cv2.setTrackbarPos('Black Lower S', 'sliders', self.black_lower[1])
        cv2.createTrackbar('Black Lower V', 'sliders', 0, 255, lambda x: self.update_black_lower_b(x))
        cv2.setTrackbarPos('Black Lower V', 'sliders', self.black_lower[2])
        cv2.createTrackbar('Black Upper H', 'sliders', 0, 255, lambda x: self.update_black_upper_r(x))
        cv2.setTrackbarPos('Black Upper H', 'sliders', self.black_upper[0])
        cv2.createTrackbar('Black Upper S', 'sliders', 0, 255, lambda x: self.update_black_upper_g(x))
        cv2.setTrackbarPos('Black Upper S', 'sliders', self.black_upper[1])
        cv2.createTrackbar('Black Upper V', 'sliders', 0, 255, lambda x: self.update_black_upper_b(x))
        cv2.setTrackbarPos('Black Upper V', 'sliders', self.black_upper[2])

    def accumulate_contours(self, contours):
        dest = []
        if contours:
            dest += contours

        return dest

    # @timeit('assign_color_to_pen()')
    def detect(self, frame, ids_and_points):
        print(ids_and_points)
        id_pen_association = {}

        if self.start:
            self.start = False
            if DEBUG:
                self.setup_sliders()

        if DEBUG:
            print(f"self.red_lower = np.array({[v for v in self.red_lower]})")
            print(f"self.red_upper = np.array({[v for v in self.red_upper]})")
            print(f"self.green_lower = np.array({[v for v in self.green_lower]})")
            print(f"self.green_upper = np.array({[v for v in self.green_upper]})")
            print(f"self.blue_lower = np.array({[v for v in self.blue_lower]})")
            print(f"self.blue_upper = np.array({[v for v in self.blue_upper]})")
            print(f"self.white_lower = np.array({[v for v in self.white_lower]})")
            print(f"self.white_upper = np.array({[v for v in self.white_upper]})")
            print(f"self.black_lower = np.array({[v for v in self.black_lower]})")
            print(f"self.black_upper = np.array({[v for v in self.black_upper]})")
            print("----")

        width, height = 1280, 720
        crop_radius = int((3.84 * (97.5 + 40)) / (3840 / width)) # 3.84 px per mm; 97.5 mm pen white part; 40mm ping pong ball diameter
        crops = []

        for i, (_id, x, y) in enumerate(ids_and_points):
            red, green, blue = False, False, False

            x = int(x / 3840 * width)
            y = int(y / 2160 * height)

            up, bottom = max(0, y - crop_radius), min(y + crop_radius, height)
            left, right = max(0, x - crop_radius), min(width, x + crop_radius)

            crop = frame[up:bottom, left:right]
            crops.append([_id, crop, left, up])

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hsv = cv2.erode(hsv, np.ones((5, 5), np.uint8))
            hsv = cv2.dilate(hsv, np.ones((5, 5), np.uint8))

            r = cv2.inRange(hsv, self.red_lower, self.red_upper)
            g = cv2.inRange(hsv, self.green_lower, self.green_upper)
            b = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

            # w = cv2.inRange(hsv, self.white_lower, self.white_upper)
            # black = cv2.inRange(hsv, self.black_lower, self.black_upper)

            if DEBUG:
                cv2.imshow(f"Crop{i} R", r)
                cv2.imshow(f"Crop{i} G", g)
                cv2.imshow(f"Crop{i} B", b)
                # cv2.imshow(f"Crop{i} W", w)
                # cv2.imshow(f"Crop{i} BLACK", black)

            rcontours, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, up))
            gcontours, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, up))
            bcontours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, up))
            # wcontours, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, up))
            # blackcontours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(left, up))

            c = rcontours
            c = c[0:5]
            contours = self.accumulate_contours(c)

            if len(contours):
                if DEBUG:
                    red_contours = np.vstack(contours)
                    col = (0, 0, 255)
                    rect = cv2.minAreaRect(red_contours)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    x, y, w, h = cv2.boundingRect(box)
                    offset = 10
                    cropbox = frame[y + offset:y + h - offset, x + offset:x + w - offset]

                    if len(cropbox):
                        hsvcropbox = cv2.cvtColor(cropbox, cv2.COLOR_BGR2HSV)

                        black = cv2.inRange(hsvcropbox, self.black_lower, self.black_upper)
                        cv2.imshow(f"Cropbox{i} Black", black)
                        blackcontours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x + offset, y + offset))

                        if len(blackcontours):
                            blconts = np.vstack(blackcontours)
                            black_rect = cv2.minAreaRect(blconts)
                            blackbox = cv2.boxPoints(black_rect)
                            blackbox = np.int0(blackbox)

                            center, _, angle = black_rect
                            angle -= 90

                            print(center, angle)
                            x2 = int(int(center[0]) + 15 * math.cos(angle * math.pi / 180))
                            y2 = int(int(center[1]) + 15 * math.sin(angle * math.pi / 180))

                            cv2.line(frame, (int(center[0]), int(center[1])), (x2, y2), (255, 255, 255), 3)

                            # cv2.drawContours(cropbox, [blackbox], -1, (0, 0, 0), 3)
                            cv2.drawContours(frame, [box], -1, col, 3)
                            # cv2.drawContours(frame, [blackbox], -1, (0, 0, 0), 3)
                            cv2.imshow(f"Cropbox{i}", cropbox)


            # red_contour_area = sum(cv2.contourArea(c) for c in rcontours)
            # green_contour_area = sum(cv2.contourArea(c) for c in gcontours)
            # blue_contour_area = sum(cv2.contourArea(c) for c in bcontours)
            #
            # if red_contour_area > green_contour_area:
            #     red = True
            #     c = rcontours
            #     contour_area = red_contour_area
            # else:
            #     green = True
            #     c = gcontours
            #     contour_area = green_contour_area
            # if contour_area < blue_contour_area:
            #     red = False
            #     green = False
            #     blue = True
            #     c = bcontours
            #
            # c = c[0:5]
            #
            # contours = self.accumulate_contours(c)
            # wconts = self.accumulate_contours(wcontours)
            # blackconts = self.accumulate_contours(blackcontours)
            #
            # if len(contours):
            #     if DEBUG:
            #         if red:
            #             red_contours = np.vstack(contours)
            #             col = (0, 0, 255)
            #             rect = cv2.minAreaRect(red_contours)
            #         if green:
            #             green_contours = np.vstack(contours)
            #             col = (0, 255, 0)
            #             rect = cv2.minAreaRect(green_contours)
            #         if blue:
            #             blue_contours = np.vstack(contours)
            #             col = (255, 0, 0)
            #             rect = cv2.minAreaRect(blue_contours)
            #
            #         box = cv2.boxPoints(rect)
            #         box = np.int0(box)
            #
            #         if len(wconts):
            #             wbox = np.int0(cv2.boxPoints(cv2.minAreaRect(np.vstack(wconts))))
            #             cv2.drawContours(crop, [wbox], -1, (255, 255, 255), 3)
            #
            #         if len(blackconts):
            #             blackbox = np.int0(cv2.boxPoints(cv2.minAreaRect(np.vstack(blackconts))))
            #             cv2.drawContours(crop, [blackbox], -1, (0, 0, 0), 3)
            #
            #         cv2.drawContours(frame, [box], -1, col, 3)
            #         cv2.drawContours(crop, [box], -1, col, 3)
            #
            #         cv2.imshow(f"Crop{i}", crop)

                if red:
                    id_pen_association[_id] = {"color": "r"}
                if green:
                    id_pen_association[_id] = {"color": "g"}
                if blue:
                    id_pen_association[_id] = {"color": "b"}

        if DEBUG:
            cv2.imshow(f"F", frame)
            print(id_pen_association)

        return id_pen_association