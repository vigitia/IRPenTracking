import cv2
import time

CHESSBOARD_SQUARES = (9, 6)

MIN_TIME_BETWEEN_FRAMES_SEC = 3


class CalibrationImageCaptureService:

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    picture_index_for_camera = {}
    last_saved_frame_timestamps = {}

    def __init__(self, camera_names):
        for camera_name in camera_names:
            self.last_saved_frame_timestamps[camera_name] = 0
            self.picture_index_for_camera[camera_name] = 0

    def collect_calibration_image(self, image, camera_name):
        if image is not None:
            # TODO: Check if getting the grey frame is necessary
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image

            ret, corners = cv2.findChessboardCorners(image, CHESSBOARD_SQUARES, None)

            if ret:
                now = time.time()
                if now - self.last_saved_frame_timestamps[camera_name] > MIN_TIME_BETWEEN_FRAMES_SEC:
                    self.last_saved_frame_timestamps[camera_name] = now
                    cv2.imwrite('calibration_images/{}/{}.png'.format(camera_name, self.picture_index_for_camera[camera_name]), image)
                    print('Saved calibration image {}.png for {}'.format(self.picture_index_for_camera[camera_name], camera_name))
                    print('\a')
                    self.picture_index_for_camera[camera_name] += 1
                    corners2 = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(image, CHESSBOARD_SQUARES, corners2, ret)

        return self.picture_index_for_camera[camera_name]




# def main():
#     table_extraction_service = TableExtractionService()
#
#     camera = RealsenseD435Camera()
#     camera.init_video_capture()
#     camera.start()
#
#     i = -1
#     last_corners = []
#
#     def lists_are_equal(u, v):
#         if len(u) != len(v):
#             return False
#
#         u.sort()
#         v.sort()
#
#         for k in range(len(u)):
#             if u[k][0][0] != v[k][0][0] or u[k][0][1] != v[k][0][1]:
#                 return False
#
#         return True
#
#     while True:
#         color_image, _ = camera.get_frames()  # Get frames from cameras
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#         if color_image is not None:
#             #color_image = table_extraction_service.extract_table_area(color_image)
#             gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#
#             # ret, gray = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
#
#             #vis = cv2.resize(gray, (int(1920), int(1080)), interpolation=cv2.INTER_AREA)
#
#             cv2.imshow("Gray", gray)
#
#             ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SQUARES, None)
#
#             if ret:
#                 # if not lists_are_equal(corners, last_corners):
#                 #     last_corners = corners
#                 i += 1
#                 cv2.imwrite(f"calibration_images/{i}.png", gray)
#                 corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#                 cv2.drawChessboardCorners(color_image, CHESSBOARD_SQUARES, corners2, ret)
#
#
#                     # cv2.imshow("TEST", img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             exit(0)
#
#
# if __name__ == '__main__':
#     main()
