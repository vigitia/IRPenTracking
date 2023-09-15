import numpy as np
import cv2
import glob
import skimage

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

CAMERA_WIDTH = 1920  # 848
CAMERA_HEIGHT = 1200  # 480

factor_width = WINDOW_WIDTH / CAMERA_WIDTH
factor_height = WINDOW_HEIGHT / CAMERA_HEIGHT


# TODO: Fix possible offset
def find_pen_position_subpixel_crop(roi, center_original):
    w = roi.shape[0]
    h = roi.shape[1]
    # print('1', ir_image.shape)
    # center_original = (coords_original[0] + w/2, coords_original[1] + h/2)

    new_w = int(w * factor_width)
    new_h = int(h * factor_height)
    top_left_scaled = (center_original[0] * factor_width - new_w / 2,
                       center_original[1] * factor_height - new_h / 2)

    # print(w, h, factor_w, factor_h, new_w, new_h, center_original, top_left_scaled)

    # cv2.imshow('roi', roi)
    # TODO:
    # print('2', ir_image_grey.shape)
    # Set all pixels
    _, thresh = cv2.threshold(roi, np.max(roi) - 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)

    # TODO: resize only cropped area
    # thresh_large = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    thresh_large = skimage.transform.resize(thresh, (new_w, new_h), mode='edge', anti_aliasing=False,
                                            anti_aliasing_sigma=None, preserve_range=True, order=0)
    # thresh_large_preview = cv2.cvtColor(thresh_large.copy(), cv2.COLOR_GRAY2BGR)

    contours = cv2.findContours(thresh_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    min_radius = thresh_large.shape[0]
    smallest_contour = contours[0]
    min_center = (0, 0)

    # print(len(contours))
    # Find the smallest contour if there are multiple (we want to find the pen tip, not its light beam
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius:
            min_radius = radius
            smallest_contour = contour
            min_center = (round(x), round(y))

    min_radius = 1 if min_radius < 1 else min_radius

    if len(smallest_contour) < 4:
        cX, cY = min_center
        # print('small contour')
        # TODO: HOW TO HANDLE SMALL CONTOURS?
    else:
        # Find the center of the contour using OpenCV Moments

        M = cv2.moments(smallest_contour)
        # calculate x,y coordinate of center
        try:
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])

            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

        except Exception as e:
            print(e)
            print('Error in find_pen_position_subpixel_crop()')
            print('Test', min_radius, min_center, len(smallest_contour))
            # time.sleep(5)
            cX = 0
            cY = 0

    # print(cX, cY)

    # position = (int(top_left_scaled[0] + cX), int(top_left_scaled[1] + cY))
    position = (top_left_scaled[0] + cX, top_left_scaled[1] + cY)

    # thresh_large_preview = cv2.drawContours(thresh_large_preview, [smallest_contour], 0, (0, 0, 255), 1)
    # thresh_large_preview = cv2.circle(thresh_large_preview, min_center, round(min_radius), (0, 255, 0), 1)
    # thresh_large_preview = cv2.circle(thresh_large_preview, min_center, 0, (0, 255, 0), 1)
    # thresh_large_preview = cv2.circle(thresh_large_preview, (round(cX), round(cY)), 0, (0, 128, 128), 1)
    # cv2.imshow('thresh_large', thresh_large_preview)

    # print(min_radius, cX, cY, position)

    return position, min_radius


files = glob.glob('rois_draw/*')

for file in files:

    img = cv2.imread(file)

    unique, counts = np.unique(img, return_counts=True)

    print('-------------------')
    print(np.asarray((unique, counts)).T)



    roi_preview = img.copy()
    roi_preview = cv2.resize(roi_preview, (960, 960), interpolation=cv2.INTER_AREA)

    # show image
    cv2.imshow('roi_preview', roi_preview)
    cv2.waitKey(1000)


cv2.destroyAllWindows()