
import numpy as np
from tensorflow import keras
from tflite import LiteModel

MODEL_PATH = 'cnn'  # Put the folder path here for the desired cnn

# Allowed states for CNN prediction
STATES = ['draw', 'hover', 'hover_far', 'undefined']


class IRPenCNN:
    keras_lite_model = None

    def __init__(self):
        self.__init_keras()

    def __init_keras(self):
        keras.backend.clear_session()
        self.keras_lite_model = LiteModel.from_keras_model(keras.models.load_model(MODEL_PATH))
        # self.keras_lite_model = keras.models.load_model(MODEL_PATH)

    # @timeit('Predict')
    def predict(self, img):
        # if len(image.shape) == 3:
        #     print(image[10,10,:])
        #     image = image[:, :, :2]
        #     print(image[10, 10, :], 'after')
        # image = image.astype('float32') / 255
        # if len(image.shape) == 3:
        #     image = image.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 2)
        # else:
        #     image = image.reshape(-1, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 1)

        img_reshaped = img.reshape(-1, img.shape[0], img.shape[1], 1)
        prediction = self.keras_lite_model.predict(img_reshaped)
        if not prediction.any():
            return STATES[-1], 0
        state = STATES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # if ACTIVE_LEARNING_COLLECTION_MODE:
        #     if state != self.active_learning_state:
        #         cv2.imwrite(f'{TRAIN_PATH}/{TRAIN_STATE}/{TRAIN_STATE}_{self.active_learning_counter}.png', image)
        #         print(f'saving frame {self.active_learning_counter}')
        #         self.active_learning_counter += 1

        # print(state)
        if state == 'hover_far':
            state = 'hover'
        return state, confidence
