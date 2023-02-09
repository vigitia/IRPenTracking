import os
import sys
import time
import cv2

# Enable to collect training images if you want to retrain the CNN (see train_network.ipynb)
ACTIVE_LEARNING_COLLECTION_MODE = False

TRAIN_PATH = 'training_images/2022-11-22'
TRAIN_IMAGE_COUNT = 3000

CATEGORY = 'hover_far'  # draw, hover_close or hover far
TRAINING_ROUND = 0  # Increment after each training round


NUM_WAIT_FRAMES = 3


class TrainingImagesCollector:

    train_state = ''

    # Keeps track of the number of saved images
    saved_image_counter = 0
    num_saved_images_cam_0 = 0
    num_saved_images_cam_1 = 0

    def __init__(self, ir_pen, cam_exposure_time_microseconds, camera_gain):

        self.ir_pen = ir_pen
        self.train_state = '{}_{}_{}_{}'.format(CATEGORY, TRAINING_ROUND, cam_exposure_time_microseconds, camera_gain)

        if not os.path.exists(os.path.join(TRAIN_PATH, self.train_state)):
            os.makedirs(os.path.join(TRAIN_PATH, self.train_state))
        else:
            print('WARNING: FOLDER ALREADY EXISTS. PLEASE EXIT TRAINING MODE IF THIS WAS AN ACCIDENT')
            time.sleep(100000)

    def save_training_images(self, camera_frames):
        for i, frame in enumerate(camera_frames):

            # TODO: Get here all spots and not just one
            pen_event_roi, brightest, (x, y) = self.ir_pen.crop_image_old(frame)

            if pen_event_roi is not None:
                self.__save_training_image(pen_event_roi, (x, y), i)

    def __save_training_image(self, img, pos, camera_id):
        if self.saved_image_counter == 0:
            print('Starting in 5 Seconds')
            # TODO: Replace sleep here with a timestamp check
            time.sleep(5)

        self.saved_image_counter += 1
        if self.saved_image_counter % NUM_WAIT_FRAMES == 0:

            cv2.imwrite(f'{TRAIN_PATH}/{self.train_state}/{self.train_state}_{int(self.saved_image_counter / NUM_WAIT_FRAMES)}_{pos[0]}_{pos[1]}.png', img)

            print(f'saving frame {int(self.saved_image_counter / NUM_WAIT_FRAMES)}/{TRAIN_IMAGE_COUNT} from camera {camera_id}')

            if camera_id == 0:
                self.num_saved_images_cam_0 += 1
            elif camera_id == 1:
                self.num_saved_images_cam_1 += 1

        if self.saved_image_counter / NUM_WAIT_FRAMES >= TRAIN_IMAGE_COUNT:
            print('FINISHED COLLECTING TRAINING IMAGES. Saved {} images from cam 0 and {} images from cam 1'.format(
                self.num_saved_images_cam_0, self.num_saved_images_cam_1))
            time.sleep(10)
            sys.exit(0)
