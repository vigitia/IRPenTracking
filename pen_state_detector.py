import numpy as np
import tensorflow as tf
from tensorflow import keras
from tflite import LiteModel

IMG_SIZE = 48
STATES = ['draw', 'hover', 'direct', 'undefined']

keras.backend.clear_session()

model = keras.models.load_model('evaluation/hover_predictor_binary_7')
#model = keras.models.load_model('evaluation/hover_predictor_three_1')
litemodel = LiteModel.from_keras_model(model)

def predict(img):
    img = img.astype('float32') / 255
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = litemodel.predict(img)
    if not prediction.any():
        return STATES[-1], 0
    state = STATES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return state, confidence
