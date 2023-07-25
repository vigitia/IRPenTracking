import os
import cv2
import random
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from keras.utils import to_categorical

MIN_BRIGHTNESS_PREDICTION = 50

# Removes all Pixels from the image that are darker than MIN_EXPECTED_BACKGROUND_BRIGHTNESS
REMOVE_BACKGROUND_NOISE = True
MIN_EXPECTED_BACKGROUND_BRIGHTNESS = 10


class PrepareDataset:

    def __init__(self):
        pass

    def get_dataset(self, target_folder, img_size):
        draw_paths, hover_paths = self.get_image_folders(target_folder)
        images_draw, images_hover = self.read_images(draw_paths, hover_paths, img_size)
        images_draw_test, images_hover_test, \
            images_draw_train, images_hover_train = self.split_train_test(images_draw, images_hover)

        self.preview_data(images_draw_test, 'DRAW')
        self.preview_data(images_hover_test, 'HOVER')

        train_X, train_label, test_X, test_label = self.prepare_final_dataset(images_draw_test, images_hover_test,
                                                                              images_draw_train, images_hover_train)

        return train_X, train_label, test_X, test_label

    def get_image_folders(self, target_folder):

        print('Step 1: Get image folders')

        draw_paths = []
        hover_paths = []

        subfolders = (next(os.walk(target_folder))[1])

        for subfolder_name in subfolders:
            if 'draw' in subfolder_name:
                draw_paths.append(os.path.join(target_folder, subfolder_name))
            elif 'hover' in subfolder_name:
                hover_paths.append(os.path.join(target_folder, subfolder_name))
            else:
                print('ERROR: Malformed folder name')

        print('All folders with "draw"  images: ', draw_paths)
        print('All folders with "hover" images: ', hover_paths)
        print('')

        return draw_paths, hover_paths

    def read_images(self, draw_paths, hover_paths, img_size):
        images_draw = []
        images_hover = []

        for path in draw_paths:
            images_draw += self.read_images_mono(path, img_size)

        for path in hover_paths:
            images_hover += self.read_images_mono(path, img_size)

        # Shuffle images
        random.shuffle(images_draw)
        random.shuffle(images_hover)

        return images_draw, images_hover

    def read_images_mono(self, path, img_size):
        files = os.listdir(path)
        files = [file for file in files if file[-4:] == ".png"]

        # print('Read {} files from "{}"'.format(len(files), path))

        images = []
        # coords = []  # TODO: Implement if needed

        skip_count = 0

        for file in tqdm(files):
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if image.shape == (img_size, img_size) and np.max(
                    image) > MIN_BRIGHTNESS_PREDICTION:

                # Remove Background "noise"
                if REMOVE_BACKGROUND_NOISE:
                    image[image < MIN_EXPECTED_BACKGROUND_BRIGHTNESS] = 0

                images.append(image)
            else:
                skip_count += 1

        if skip_count > 0:
            print('Skipped {} images in folder {} because they were too dark'.format(skip_count, path))

        return images

    def data_augmentation(self, images):
        # TODO: Use exisiting data augmentation function from keras here!
        result = []
        for img in images:
            rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
            rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            result.append(img)  # Add original image
            result.append(cv2.flip(img, flipCode=0))  # Add flipped original
            result.append(rotate_90)
            result.append(cv2.flip(rotate_90, flipCode=0))
            result.append(rotate_180)
            result.append(cv2.flip(rotate_180, flipCode=0))
            result.append(rotate_270)
            result.append(cv2.flip(rotate_270, flipCode=0))

            # TODO: Add brightness modifications back if wanted
            # for i in range(3):
            #     tmp = cv2.rotate(img, i)  # Rotate: ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE
            #     for b in range(10, 11):
            #         tmp2 = np.clip(tmp * (b / 10), 0, 255)
            #         result.append(tmp2)
            #         result.append(cv2.flip(tmp2, flipCode=0))
        return result

        # data_augmentation_new = tf.keras.Sequential([
        #     RandomFlip("horizontal_and_vertical"),
        #     # RandomRotation(0.25),
        #     RandomBrightness(0.1),
        # ])

    def split_train_test(self, images_draw, images_hover):
        images_draw_test = []
        images_hover_test = []
        images_draw_train = []
        images_hover_train = []

        SPLIT_NUMBER = 2

        for i, img in enumerate(images_draw):
            if i % SPLIT_NUMBER == 0:
                images_draw_test.append(img)
            else:
                images_draw_train.append(img)

        for i, img in enumerate(images_hover):
            if i % SPLIT_NUMBER == 0:
                images_hover_test.append(img)
            else:
                images_hover_train.append(img)

        return images_draw_test, images_hover_test, images_draw_train, images_hover_train

    def prepare_final_dataset(self, images_draw_test, images_hover_test, images_draw_train, images_hover_train):

        # Only apply data augmentation to training data, NOT to test data
        DATA_AUGMENTATION = False
        if DATA_AUGMENTATION:
            images_draw_train = self.data_augmentation(images_draw_train)
            images_hover_train = self.data_augmentation(images_hover_train)

        # Shuffle images
        random.shuffle(images_draw_train)
        random.shuffle(images_hover_train)

        # Combine
        images_train = images_draw_train + images_hover_train
        images_test = images_draw_test + images_hover_test

        labels_train = [0] * len(images_draw_train) + [1] * len(images_hover_train)
        labels_test = [0] * len(images_draw_test) + [1] * len(images_hover_test)

        print(' ')
        print('We have {} images in the training dataset and {} images in the test dataset'.format(len(images_train),
                                                                                                   len(images_test)))
        print('The training dataset consists of {} draw images and {} hover images'.format(len(images_draw_train),
                                                                                           len(images_hover_train)))
        print(' ')

        train_X = np.array(images_train)
        train_X = train_X.astype('float32')
        train_X = train_X / 255.

        test_X = np.array(images_test)
        test_X = test_X.astype('float32')
        test_X = test_X / 255.

        train_label = to_categorical(np.array(labels_train))
        test_label = to_categorical(np.array(labels_test))

        # unique, counts = np.unique(test_label, return_counts=True)
        # print(dict(zip(unique, counts)))
        #
        # unique, counts = np.unique(train_label, return_counts=True)
        # print(dict(zip(unique, counts)))
        #
        # l = [0, 0]
        # for i in test_label:
        #     # print(i)
        #     l[int(np.argmax(i))] += 1
        # print(l)

        return train_X, train_label, test_X, test_label

    def preview_data(self, images, image_type):
        ROWS = 4
        COLUMS = 5

        image_list = []
        for i in range(ROWS * COLUMS):
            image = random.sample(images, 1)[0]
            image_list.append(image)
            # histogram, bin_edges = np.histogram(image, bins=256, range=(1, 255))
            # image_list.append([histogram, bin_edges, image])

        fig = plt.figure(figsize=(16, 16))
        plt.title(f'Examples for {image_type}')
        plt.axis('off')
        grid = ImageGrid(fig, 111, nrows_ncols=(COLUMS, ROWS), axes_pad=0.1, )

        for ax, im in zip(grid, image_list):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, 'gray')


if __name__ == '__main__':
    IMG_SIZE = 48  # The expected size of the input images

    TARGET_FOLDER = 'training_images/2023-02-24'  # The folder containing the new images

    prepare_dataset = PrepareDataset()

    train_X, train_label, test_X, test_label = prepare_dataset.get_dataset(TARGET_FOLDER, IMG_SIZE)

