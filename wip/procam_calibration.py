import os
import os.path
import glob
import cv2
import numpy as np
import json

PROJECTOR_RESOLUTION = (2160, 3840)
CHESSBOARD_CORNERS = (11, 7)
CHESSBOARD_SQUARE_WIDTH_MM = 28
GRAYCODE_STEP = 1
BLACK_THRESHOLD = 40
WHITE_THRESHOLD = 5

IMAGES_PATH = 'wip/projector_camera_calibration/captures'

# TODO: Source

def main():
    image_parent_folders = sorted(glob.glob(os.path.join(IMAGES_PATH, 'capture_*')))

    file_paths_camera_0 = []
    file_paths_camera_1 = []
    for image_folder in image_parent_folders:
        filename_camera_0 = sorted(glob.glob(image_folder + '/camera0/capture_*'))
        file_paths_camera_0.append(filename_camera_0)
        filename_camera_1 = sorted(glob.glob(image_folder + '/camera1/capture_*'))
        file_paths_camera_1.append(filename_camera_1)

    print(file_paths_camera_0)
    print(file_paths_camera_1)

    camP = None
    cam_dist = None
    # path, ext = os.path.splitext(camera_param_file)
    # if(ext == ".json"):
    #     camP,cam_dist = loadCameraParam(camera_param_file)
    #     print('load camera parameters')
    #     print(camP)
    #     print(cam_dist)

    calibrate(image_parent_folders, file_paths_camera_0, camP, cam_dist)


def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))


def loadCameraParam(json_file):
    with open(json_file, 'r') as f:
        param_data = json.load(f)
        P = param_data['camera']['P']
        d = param_data['camera']['distortion']
        return np.array(P).reshape([3, 3]), np.array(d)


def calibrate(image_parent_folders, file_paths_camera_n, camP=None, camD=None):
    objps = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
    objps[:, :2] = CHESSBOARD_SQUARE_WIDTH_MM * np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)

    print('Calibrating ...')
    gc_height = int((PROJECTOR_RESOLUTION[0] - 1) / GRAYCODE_STEP) + 1
    gc_width = int((PROJECTOR_RESOLUTION[1] - 1) / GRAYCODE_STEP) + 1

    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(BLACK_THRESHOLD)
    graycode.setWhiteThreshold(WHITE_THRESHOLD)

    cam_shape = cv2.imread(file_paths_camera_n[0][0], cv2.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    print('  patch size :', patch_size_half * 2 + 1)

    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(image_parent_folders, file_paths_camera_n):
        print('  checking \'' + dname + '\'')
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            print('Error : invalid number of images in \'' + dname + '\'')
            return None

        imgs = []
        for fname in gc_filenames:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if cam_shape != img.shape:
                print('Error : image size of \'' + fname + '\' is mismatch')
                return None
            imgs.append(img)
        black_img = imgs.pop()
        white_img = imgs.pop()

        cv2.imshow('white', white_img)
        cv2.imshow('black', black_img)

        cv2.waitKey(10000)

        res, cam_corners = cv2.findChessboardCorners(white_img, CHESSBOARD_CORNERS)
        if not res:
            print('Error : chessboard was not found in \'' +
                  gc_filenames[-2] + '\'')
            return None
        else:
            print('Found chessboard')

        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        # viz_proj_points = np.zeros(proj_shape, np.uint8)

        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if int(white_img[y, x]) - int(black_img[y, x]) <= BLACK_THRESHOLD:
                        continue
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(GRAYCODE_STEP * np.array(proj_pix))
            if len(src_points) < patch_size_half ** 2:
                print('    Warning : corner', c_x, c_y, 'was skiped because decoded pixels were too few (check your images and thresholds)')
                continue
            h_mat, inliers = cv2.findHomography(np.array(src_points), np.array(dst_points))
            point = h_mat @ np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2] / point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
            # viz_proj_points[int(round(point_pix[1])),
            #                 int(round(point_pix[0]))] = 255

        if len(proj_corners) < 3:
            print('Error : too few corners were found in \'' + dname + '\' (less than 3)')
            return None
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
        # cv2.imwrite('visualize_corners_projector_' +
        #             str(cnt) + '.png', viz_proj_points)
        # cnt += 1

    print('Initial solution of camera\'s intrinsic parameters')
    cam_rvecs = []
    cam_tvecs = []

    if (camP is None):
        ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
        print('  RMS :', ret)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            ret, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camP, camD)
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            print('  RMS :', ret)
        cam_int = camP
        cam_dist = camD
    print('  Intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print()

    print('Initial solution of projector\'s parameters')
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, PROJECTOR_RESOLUTION, None, None, None, None)
    print('  RMS :', ret)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print()

    print('=== Result ===')
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None)
    print('  RMS :', ret)
    print('  Camera intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Camera distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print('  Projector intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Projector distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print('  Rotation matrix / translation vector from camera to projector')
    print('  (they translate points from camera coord to projector coord) :')
    printNumpyWithIndent(cam_proj_rmat, '    ')
    printNumpyWithIndent(cam_proj_tvec, '    ')
    print()

    fs = cv2.FileStorage('calibration_result.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('img_shape', cam_shape)
    fs.write('rms', ret)
    fs.write('cam_int', cam_int)
    fs.write('cam_dist', cam_dist)
    fs.write('proj_int', proj_int)
    fs.write('proj_dist', proj_dist)
    fs.write('rotation', cam_proj_rmat)
    fs.write('translation', cam_proj_tvec)
    fs.release()


if __name__ == '__main__':
    main()
