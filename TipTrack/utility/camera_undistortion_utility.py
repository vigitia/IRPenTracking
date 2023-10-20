
import os
import cv2

CALIBRATION_DATA_PATH = os.path.join(os.getcwd(), 'TipTrack', 'config')  # Specify location where the calibration file is saved


class CameraUndistortionUtility:
    """  CameraUndistortionUtility

        This Utility class will help you to get all necessery data (especially the undistortion and rectification
        transformation maps to undistort and calibrate camera frames


    """

    def __init__(self, target_frame_width, target_frame_height):
        self.target_frame_width = target_frame_width
        self.target_frame_height = target_frame_height

    def get_camera_undistort_rectify_maps(self, camera_name):
        camera_matrix, dist_matrix = self.__load_intrinsic_camera_calibration_data(camera_name)

        maps = self.__generate_maps(camera_matrix, dist_matrix)

        return maps

    def __generate_maps(self, camera_matrix, dist_matrix):
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_matrix, None, None,
                                                 (self.target_frame_width, self.target_frame_height), cv2.CV_32FC1)

        return [map1, map2]

    @staticmethod
    def __load_intrinsic_camera_calibration_data(camera_name):
        try:
            cv_file = cv2.FileStorage(os.path.join(CALIBRATION_DATA_PATH, camera_name), cv2.FILE_STORAGE_READ)
            camera_matrix = cv_file.getNode('K').mat()
            dist_matrix = cv_file.getNode('D').mat()

            cv_file.release()
            return camera_matrix, dist_matrix
        except Exception as e:
            print('[CameraUndistortionUtility]: Error in load_intrinsic_camera_calibration_data():', e)
            print('[CameraUndistortionUtility]: Cant load calibration data for camera {}'.format(camera_name))
