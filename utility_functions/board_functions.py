import random

import cv2
import numpy as np


def index_to_board_string(y, x):
    columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
    x = columns[x]
    return 8 - y, x


def class_from_prediction(board_array):
    CLASS_ARR = ["HB", "KB", "QB", "RB", "PB", "BB", "E", "BR", "KR", "HR", "RR", "PR"]
    predicted_class = map(lambda x: CLASS_ARR[x], list(board_array))
    return predicted_class


def get_random_free_space(board_array):
    occupied = True
    rand_x = None
    rand_y = None
    while occupied:
        rand_x = random.randint(0, 7)
        rand_y = random.randint(0, 7)
        occupied = bool(board_array[rand_y][rand_x])
    return index_to_board_string(rand_y, rand_x)


def get_random_pickup_space(board_array):
    free = True
    rand_x = None
    rand_y = None
    while free:
        rand_x = random.randint(0, 7)
        rand_y = random.randint(0, 7)
        free = not bool(board_array[rand_y][rand_x])
    return index_to_board_string(rand_y, rand_x)


def predict_chesspieces(model, field, reshape: tuple = (8, 8)):
    flattened_list = [element for sublist in field for element in sublist]
    x = np.array(
        [
            np.array(cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA))
            for image in flattened_list
        ]
    )

    normalized_image_array = (x.astype(np.float32) / 127.0) - 1

    prediction = model.predict(normalized_image_array)

    board_array = np.reshape(np.argmax(prediction, axis=1), reshape)
    return board_array
