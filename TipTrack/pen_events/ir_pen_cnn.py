
import os
import datetime
from threading import Lock

from tensorflow import keras, lite
import numpy as np


MODEL_PATH = 'cnn/models/TipTrack_CNN'  # 'model_2023_026'  # 'cnn'  # Put the folder path here for the desired cnn

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
    def get_prediction(self, img):
        img_reshaped = img.reshape(-1, img.shape[0], img.shape[1], 1)

        # use predict() for safe and less performant
        # use predict_unsafe() for best performance, but we are not sure what could happen in the worst case
        prediction = self.keras_lite_model.predict_unsafe(img_reshaped)
        # prediction = self.keras_lite_model.predict(img_reshaped)

        if not prediction.any():
            print('[IRPenCNN]: Prediction failed! ROI shape:', img.shape)
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


# source: Michael Wurm, 2019 on medium
# https://micwurm.medium.com/using-tf-lite-to-speed-up-predictions-a3954886eb98
class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        self.working = False
        self.lock = Lock()

    # @timeit('Predict tflite')
    def predict(self, inp):
        self.lock.acquire()

        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)

        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]

        self.lock.release()
        return out

    # TODO: rename
    def predict_unsafe(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)

        # TODO: this is probably very very dangerous
        try:
            interpreter = self.interpreter  # copy.deepcopy(self.interpreter)

            for i in range(count):
                interpreter.set_tensor(self.input_index, inp[i:i + 1])
                interpreter.invoke()
                out[i] = interpreter.get_tensor(self.output_index)[0]

        except Exception as e:
            pass
            # TODO: Andis dangerous bit. Turns out = is not a deep copy.
            # print('Andis dangerous bit. Turns out = is not a deep copy.', e)

        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]