import cv2
import numpy as np
import glob

# Based on: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
import yaml

CHESSBOARD_SQUARES = (9, 6)


# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class CameraCalibrationService:

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.object_points = np.zeros((CHESSBOARD_SQUARES[0] * CHESSBOARD_SQUARES[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:CHESSBOARD_SQUARES[0], 0:CHESSBOARD_SQUARES[1]].T.reshape(-1, 2)

    def calibrate_cameras(self, cameras):

        print('STARTING CAMERA CALIBRATION_PROCESS')

        for camera in cameras:

            image_path = "calibration_images/{}/*.png".format(camera)

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.

            # get the paths to all images in path
            images = glob.glob(image_path)

            for filename in images:
                image = cv2.imread(filename)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                # _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

                # Find the Chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SQUARES, None)

                # If found, add object points, image points (after refining them)
                if ret:
                    objpoints.append(self.object_points)
                    # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                    imgpoints.append(corners)

                    cv2.drawChessboardCorners(image, CHESSBOARD_SQUARES, corners2, ret)
                    cv2.imshow('CALIBRATION', image)

                    key = cv2.waitKey(200)
                    if key & 0xFF == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        break

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            self.save_coefficients(mtx, dist, camera)

            # # transform the matrix and distortion coefficients to writable lists
            # data = {'camera_matrix': np.asarray(mtx).tolist(),
            #         'dist_coeff': np.asarray(dist).tolist()}
            #
            # # # Save to a yaml file
            # # with open("calibration_matrix_{}.yaml".format(camera), "w") as f:
            # #     yaml.dump(data, f)

    def save_coefficients(self, mtx, dist, camera):
        # Save the camera matrix and the distortion coefficients to given path/file.
        cv_file = cv2.FileStorage('{}.yml'.format(camera), cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)
        cv_file.release()
