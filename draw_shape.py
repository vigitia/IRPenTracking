import cv2
import numpy as np
import sys
from svgpathtools import *

class ShapeCreator:
    LINE = 0
    DASH = 1
    DOT = 2

    def __init__(self, w, h):
        self.display_width = w
        self.display_height = h
        self.resolution = 0.3

    def draw_shape(self, path, coords, width, thickness, style=LINE, color=(255, 255, 255), resolution=0):
        img = np.zeros((self.display_height, self.display_width, 3))
        if resolution == 0:
            resolution = self.resolution
        paths = self._shape_from_svg(path, coords, width, resolution)

        if style == ShapeCreator.LINE:
            for points in paths:
                cv2.polylines(img, [points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        elif style == ShapeCreator.DASH:
            for points in paths:
                points_1 = points[::20]
                points_2 = points[10::20]
                for i in range(len(points_1) - 1):
                    p1 = points_1[i]
                    p2 = points_2[i]
                    img = cv2.line(img, (points_1[i][0], points_1[i][1]), (points_2[i][0], points_2[i][1]), color, thickness=thickness)
        elif style == ShapeCreator.DOT:
            for points in paths:
                points_1 = points[::10]
                points_2 = points[5::10]
                for i in range(len(points_1) - 1):
                    p1 = points_1[i]
                    p2 = points_2[i]
                    img = cv2.circle(img, (p1[0], p1[1]), thickness, color, thickness=-1)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # source: https://stackoverflow.com/questions/45809719/fast-way-to-sample-geometric-points-from-svg-paths
    def _shape_from_svg(self, svg_path, coords, width, resolution=0.3):
        paths, _ = svg2paths(svg_path)
        result = []
        mins = []
        maxs = []
        for path in paths:
            points = np.array([path.point(i) for i in np.linspace(0, 1, int(path.length() * resolution))])
            points = np.stack((points.real, points.imag), -1)
            mins.append(np.min(points, axis=0))
            maxs.append(np.max(points, axis=0))

        mins = np.array(mins)
        maxs = np.array(maxs)
        org_width = (np.max(maxs, axis=0)[0] - np.min(mins, axis=0)[0])
        org_height = (np.max(maxs, axis=0)[1] - np.min(mins, axis=0)[1])
        height = (org_height / org_width) * width
        for path in paths:
            points = np.array([path.point(i) for i in np.linspace(0, 1, int(path.length() * resolution))])
            points = np.stack((points.real, points.imag), -1)
            points -= np.min(mins)
            points /= (np.max(maxs) - np.min(mins))
            points *= width
            points += (np.array(coords) - (np.array([width, height]) / 2))
            points = np.array(points, dtype='int')
            result.append(points)
        return np.array(result)

if __name__ == '__main__':
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1080

    shape_creator = ShapeCreator(WINDOW_WIDTH, WINDOW_HEIGHT)
    img = shape_creator.draw_shape('shapes/wave.svg', (500, 500), 500, 3, ShapeCreator.DASH, (255, 0, 0))

    cv2.imshow('roi', img)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
