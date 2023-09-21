from tensorflow import keras, lite

# Old Code to test this script
# TODO: Rework
if __name__ == '__main__':
    import cv2
    import os
    import random
    from matplotlib import pyplot as plt

    STATES = ['draw', 'hover', 'undefined']

    MODEL_PATH = 'evaluation/model_new_projector_5'
    keras.backend.clear_session()
    keras_lite_model = LiteModel.from_keras_model(keras.models.load_model(MODEL_PATH))
    # keras_lite_model = keras.models.load_model(MODEL_PATH)
    print(os.getcwd())
    draw_path = 'out3/2022-08-02/draw_1_400_18/'
    image_paths_draw = os.listdir(draw_path)
    hover_path = 'out3/2022-08-02/hover_close_1_400_18/'
    image_paths_hover = os.listdir(hover_path)
    correct = 0
    num_draw = 0
    correct_draw = 0
    num_hover = 0
    correct_hover = 0
    brightnesses = []

    draw_path_temp = []
    for filename in image_paths_draw:
        if '681_597' not in filename:
            draw_path_temp.append(filename)
        else:
            print('Oh no')

    image_paths_draw = draw_path_temp

    too_dark = 0
    for image in image_paths_draw:
        img = cv2.imread(draw_path + random.sample(image_paths_draw, 1)[0], cv2.IMREAD_GRAYSCALE)
        brightnesses.append(np.max(img))
        if np.max(img) < 50:
            too_dark += 1

    print('TOO DARK:', too_dark / len(image_paths_draw))

    for i in range(1000):
        condition = 'hover' if int(random.random() * 1000) % 2 == 0 else 'draw'
        # condition = 'draw'
        if condition == 'draw':
            num_draw += 1
            img = cv2.imread(draw_path + random.sample(image_paths_draw, 1)[0], cv2.IMREAD_GRAYSCALE)
            # print('Max', np.max(image))
        else:
            num_hover += 1
            img = cv2.imread(hover_path + random.sample(image_paths_hover, 1)[0], cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('test', image)
        # cv2.waitKey(0)
        img = img.reshape(-1, 48, 48, 1)
        print(condition)
        prediction = keras_lite_model.predict(img)
        print(prediction)
        state = STATES[np.argmax(prediction)]
        if condition == state:
            if condition == 'draw':
                correct_draw += 1
            if condition == 'hover':
                correct_hover += 1
            correct += 1
        # print(condition, state)
    print('----')
    print('Correct total:', correct, '%')
    print('Correct Draw:', correct_draw, ' / ', num_draw)  # int((correct_draw/num_draw)*100), '%')
    print('Correct Hover:', correct_hover, ' / ', num_hover)  # int((correct_draw/num_draw)*100), '%')
    plt.boxplot(brightnesses)  # , np.ones(len(brightnesses), np.uint8))
    plt.scatter(np.ones(len(brightnesses), np.uint8), brightnesses, alpha=0.2)
    plt.show()