
import os
import numpy as np
from tensorflow import keras
from tflite import LiteModel
import datetime

MODEL_PATH = 'cnn'  # 'model_2023_026'  # 'cnn'  # Put the folder path here for the desired cnn

# Allowed states for CNN prediction
STATES = ['draw', 'hover', 'hover_far', 'undefined']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress Tensorflow warnings


# For debugging purposes
# Decorator to print the run time of a single function
# Based on: https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            return_value = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return return_value

        return wrapper

    return timeit_decorator


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
        img_reshaped = img.reshape(-1, img.shape[0], img.shape[1], 1)

        # use predict() for safe and less performant
        # use predict_unsafe() for best performance but we are not sure what could happen in the worst case
        prediction = self.keras_lite_model.predict_unsafe(img_reshaped)

        if not prediction.any():
            print('No prediction possible!')
            return STATES[-1], 0
        state = STATES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # if ACTIVE_LEARNING_COLLECTION_MODE:
        #     if state != self.active_learning_state:
        #         cv2.imwrite(f'{TRAIN_PATH}/{TRAIN_STATE}/{TRAIN_STATE}_{self.active_learning_counter}.png', image)
        #         print(f'saving frame {self.active_learning_counter}')
        #         self.active_learning_counter += 1

        # print(state)
        # if state == 'hover_far':
        #     state = 'hover'
        return state, confidence
