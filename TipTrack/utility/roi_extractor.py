
import cv2
import numpy as np


MAX_NUMBER_OF_ROIS_PER_FRAME = 10  # Set a limit on how many ROIs to accept at the maximum for each frame

MIN_BRIGHTNESS = 50  # A spot in the camera image needs to have at least X brightness to be considered.


class ROIExtractor:

    def __init__(self, crop_image_size):
        self.crop_image_size = crop_image_size
        self.margin = int(self.crop_image_size / 2)
        # Create a black rectangle to fill the ROIs after their extraction
        self.cutout = np.zeros((self.crop_image_size, self.crop_image_size), 'uint8')

    # @timeit('get_all_rois')
    def get_all_rois(self, image):
        """ Get all ROIs

            Extract all ROIs from the current frame

        """

        # TODO: Check why we crash here sometimes
        # TODO: If you get too many ROIs, the exposure of the camera should be reduced
        # TODO: Maybe some sort of auto calibration could fix this

        rois_new = []
        roi_coords_new = []
        max_brightness_values = []

        try:
            # TODO: Try to only copy the ROI and not the entire image.
            #  Does not work because I can't set the image flag to writeable.
            image_copy = image.copy()

            for i in range(MAX_NUMBER_OF_ROIS_PER_FRAME):

                # Get max brightness value of frame and its location
                _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image_copy)

                # Stop if point is not bright enough to be considered
                if brightest < MIN_BRIGHTNESS:
                    # if i > 0:
                    #    print('Stopped after {} iterations. Brightness values {}'.format(i, max_brightness_values))
                    return rois_new, roi_coords_new, max_brightness_values

                # Cut out region of interest around brightest point in image
                img_cropped = image[max_y - self.margin: max_y + self.margin, max_x - self.margin: max_x + self.margin]

                # If the point is close to the image border, the output image will be too small
                # TODO: Improve this later. Currently no need, as long as the camera FOV is larger than the projection area.
                # Problems only appear on the image border.

                if img_cropped.shape[0] == self.crop_image_size and img_cropped.shape[1] == self.crop_image_size:

                    # image.setflags(write=1)
                    # Set all pixels in ROI to black
                    image_copy[max_y - self.margin: max_y + self.margin, max_x - self.margin: max_x + self.margin] = self.cutout
                    # image.setflags(write=0)

                    rois_new.append(img_cropped)
                    # TODO: Forward the top-left corner of the ROI here and not the brightest spot
                    roi_coords_new.append((max_x, max_y))
                    max_brightness_values.append(brightest)

                else:
                    # print('ROI shape too small')

                    # Set just this pixel to black
                    # TODO: Needs improvement, set also the surrounding area to black
                    image_copy[max_y, max_x] = np.zeros((1, 1, 1), 'uint8')

        except Exception as e:
            print('[IRPen]: ERROR in/after "self.get_all_rois(frame):"', e)
            rois_new, roi_coords_new, max_brightness_values = [], [], []

        return rois_new, roi_coords_new, max_brightness_values

    # WIP!
    # def get_all_rois(self, img, size=CROP_IMAGE_SIZE):
    #
    #     rois = []
    #     roi_coords = []
    #     max_brightness_values = []
    #
    #     for i in range(10):
    #         margin = int(size / 2)
    #         _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(img)
    #
    #         # Dead pixel fix
    #         if max_x == 46 and max_y == 565:
    #             # TODO: Find solution for dead pixel
    #             continue
    #
    #         if brightest < MIN_BRIGHTNESS:
    #             break
    #
    #         img_cropped = img[max_y - margin: max_y + margin, max_x - margin: max_x + margin].copy()
    #
    #         # print('Shape in crop 1:', img_cropped.shape, max_x, max_y)
    #
    #         if img_cropped.shape == (size, size):
    #             rois.append(img_cropped)
    #             roi_coords.append((max_x, max_y))
    #             max_brightness_values.append(int(brightest))
    #         else:
    #             print('TODO: WRONG SHAPE')
    #
    #         img.setflags(write=True)
    #         img[max_y - margin: max_y + margin, max_x - margin: max_x + margin] = 0
    #         img.setflags(write=False)
    #
    #     return rois, roi_coords, max_brightness_values

    def crop_image_old(self, image):
        margin = int(self.crop_image_size / 2)

        # Get max brightness value of frame and its location
        _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image)

        # Stop if point is not bright enough to be considered
        if brightest < MIN_BRIGHTNESS:
            return None, brightest, (max_x, max_y)

        # Cut out region of interest around brightest point in image
        img_cropped = image[max_y - margin: max_y + margin, max_x - margin: max_x + margin]

        # If the point is close to the image border, the output image will be too small
        # TODO: Improve this later. Currently no need, as long as the camera FOV is larger than the projection area.
        # Problems only appear on the image border.
        if img_cropped.shape[0] != self.crop_image_size or img_cropped.shape[1] != self.crop_image_size:
            # img_cropped, brightest, (max_x, max_y) = self.crop_image_2(image)
            print('!#-#-#-#-#-#-#-#-#')
            print('Shape in crop 1:', img_cropped.shape, max_x, max_y)
            return None, brightest, (max_x, max_y)
            # time.sleep(20)

        # img_cropped_large = cv2.resize(img_cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('large', img_cropped_large)
        return img_cropped, brightest, (max_x, max_y)

    # TODO: CHECK this function. Currently not in use
    def crop_image_2_old(self, img):
        # print('using crop_image_2() function')
        margin = int(self.crop_image_size / 2)
        brightest = int(np.max(img))
        _, thresh = cv2.threshold(img, brightest - 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print(contours)

        contours = contours[0] if len(contours) == 2 else contours[1]

        x_values = []
        y_values = []
        for cnt in contours:
            for point in cnt:
                point = point[0]
                # print(point)
                x_values.append(point[0])
                y_values.append(point[1])

        # print('x', np.max(x_values), np.min(x_values))
        # print('y', np.max(y_values), np.min(y_values))
        d_x = np.max(x_values) - np.min(x_values)
        d_y = np.max(y_values) - np.min(y_values)
        center_x = int(np.min(x_values) + d_x / 2)
        center_y = int(np.min(y_values) + d_y / 2)
        # print(center_x, center_y)

        left = np.max([0, center_x - margin])
        top = np.max([0, center_y - margin])

        # print(left, top)

        if left + self.crop_image_size >= img.shape[1]:
            # left -= (left + size - image.shape[1] - 1)
            left = img.shape[1] - self.crop_image_size - 1
        if top + self.crop_image_size >= img.shape[0]:
            # top -= (top + size - image.shape[0] - 1)
            top = img.shape[0] - self.crop_image_size - 1

        # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(image)
        img_cropped = img[top: top + self.crop_image_size, left: left + self.crop_image_size]

        print('Shape in crop 2:', img_cropped.shape, left + margin, top + margin)

        return img_cropped, np.max(img_cropped), (left + margin, top + margin)