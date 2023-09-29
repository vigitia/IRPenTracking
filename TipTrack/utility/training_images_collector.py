import os
import sys
import time
import cv2

# Enable to collect training images if you want to retrain the CNN (see train_network.ipynb)
ACTIVE_LEARNING_COLLECTION_MODE = False

TRAIN_PATH = '../../cnn/training_images/2023-02-24'
TRAIN_IMAGE_COUNT = 5000

TRAINING_CATEGORIES = ['draw', 'hover_close', 'hover_far']

CATEGORY = TRAINING_CATEGORIES[2]

TRAINING_ROUND = 2  # Increment after each training round

NUM_WAIT_FRAMES = 10


class TrainingImagesCollector:

    train_state = ''

    frame_counter = 0

    # Keeps track of the number of saved images
    saved_image_counter = 0
    num_saved_images_cam_0 = 0
    num_saved_images_cam_1 = 0

    start_timestamp = -1

    def __init__(self, ir_pen, cam_exposure_time_microseconds, camera_gain):

        self.ir_pen = ir_pen
        self.train_state = '{}_{}_{}_{}'.format(CATEGORY, TRAINING_ROUND, cam_exposure_time_microseconds, camera_gain)

        if not os.path.exists(os.path.join(TRAIN_PATH, self.train_state)):
            os.makedirs(os.path.join(TRAIN_PATH, self.train_state))
        else:
            print('WARNING: FOLDER ALREADY EXISTS. PLEASE EXIT TRAINING MODE IF THIS WAS AN ACCIDENT')
            time.sleep(10000)

    def save_training_images(self, camera_frames):
        for i, frame in enumerate(camera_frames):

            # TODO: Get here all spots and not just one
            rois_new, roi_coords_new, max_brightness_values = self.ir_pen.get_all_rois(frame)

            # TODO: Currently only saving one ROI per frame
            if len(rois_new) == 1:
                finished = self.__save_training_image(rois_new[0], roi_coords_new[0], i)

                if finished:
                    print(
                        'FINISHED COLLECTING TRAINING IMAGES. Saved {} images from cam 0 and {} images from cam 1'.format(
                            self.num_saved_images_cam_0, self.num_saved_images_cam_1))
                    time.sleep(100000)

    def __save_training_image(self, img, pos, camera_id):
        self.frame_counter += 1
        if self.start_timestamp == -1:
            self.start_timestamp = round(time.time() * 1000)
            print('Starting in 5 Seconds')

        elif round(time.time() * 1000) - self.start_timestamp > 5000:

            if self.frame_counter % NUM_WAIT_FRAMES == 0:

                self.saved_image_counter += 1
                cv2.imwrite(f'{TRAIN_PATH}/{self.train_state}/{self.train_state}_{self.saved_image_counter}_{pos[0]}_{pos[1]}.png', img)

                print(f'saving frame {self.saved_image_counter}/{TRAIN_IMAGE_COUNT} from camera {camera_id}')

                if camera_id == 0:
                    self.num_saved_images_cam_0 += 1
                elif camera_id == 1:
                    self.num_saved_images_cam_1 += 1

            if self.saved_image_counter >= TRAIN_IMAGE_COUNT:
                return True

        return False
