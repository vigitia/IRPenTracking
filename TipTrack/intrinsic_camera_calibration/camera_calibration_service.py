import cv2
import numpy as np
import glob

# Based on: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html


class CameraCalibrationService:

    def __init__(self, chessboard_squares, criteria):
        self.chessboard_squares = chessboard_squares
        self.criteria = criteria
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.object_points = np.zeros((self.chessboard_squares[0] * self.chessboard_squares[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.chessboard_squares[0], 0:self.chessboard_squares[1]].T.reshape(-1, 2)

    def calibrate_cameras(self, cameras):

        print('Staring camera calibration process')

        for camera in cameras:

            image_path = "calibration_images/Flir Blackfly S {}/*.png".format(camera)

            # Arrays to store object points and roi points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in roi plane.

            # get the paths to all images in path
            images = glob.glob(image_path)

            image_size = None

            for filename in images:
                image = cv2.imread(filename)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()

                if image_size is None:
                    image_size = gray.shape[::-1]
                # _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

                # Find the Chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_squares, None)

                # If found, add object points, roi points (after refining them)
                if ret:
                    objpoints.append(self.object_points)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    imgpoints.append(corners)

                    # cv2.drawChessboardCorners(image, CHESSBOARD_SQUARES, corners2, ret)
                    # cv2.imshow('CALIBRATION', image)
                    #
                    # key = cv2.waitKey(1)
                    # if key & 0xFF == ord('q') or key == 27:
                    #     cv2.destroyAllWindows()
                    #     break

            if image_size is not None and len(objpoints) > 0 and len(imgpoints) > 0:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

                self.save_coefficients(mtx, dist, camera)
            else:
                print('Error: Could not calibrate camera, no suitable frames found.')

    def save_coefficients(self, mtx, dist, camera):
        # Save the camera matrix and the distortion coefficients to given path/file.
        cv_file = cv2.FileStorage('Flir Blackfly S {}.yml'.format(camera), cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)
        cv_file.release()
