import random

import cv2
import numpy as np
from keras.models import load_model

from apiCheckApp.services import EchoService
from utility_functions.corners import get_corners
from utility_functions.image_downloader import download_image

DRAW_CORNERS = False


def slice_image(corner_coords, cam_image_in_bytes) -> list[list]:
    img_np = np.frombuffer(cam_image_in_bytes, dtype="uint8")
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    square_pixel = int(corner_coords[1][1] - corner_coords[0][1])
    ru = np.array(
        (corner_coords[0][0] + square_pixel, corner_coords[0][1] - square_pixel)
    ).astype(int)
    ld = np.array(
        (corner_coords[-1][0] - square_pixel, corner_coords[-1][1] + square_pixel)
    ).astype(int)

    min_x, max_y = ld
    max_x, min_y = ru

    img_cut = img[min_y - 10 : max_y - 10, min_x - 10 : max_x - 10]
    img_square = cv2.resize(img_cut, (square_pixel * 8, square_pixel * 8))

    field_images = [
        [
            img_square[row : row + square_pixel, column : column + square_pixel, :]
            for column in range(0, img_square.shape[1], square_pixel)
        ]
        for row in range(0, img_square.shape[0], square_pixel)
    ]

    return field_images


def class_from_prediction(board_array):
    CLASS_ARR = ["HB", "KB", "QB", "RB", "PB", "BB", "E", "BR", "KR", "HR", "RR", "PR"]
    predicted_class = map(lambda x: CLASS_ARR[x], list(board_array))
    return predicted_class


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


def index_to_board_string(y, x):
    columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
    x = columns[x]
    return 8 - y, x


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


if __name__ == "__main__":
    image_bytes = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image_bytes)
    model = load_model("models/blue_red_model_200.h5")
    board_array = predict_chesspieces(model, sliced_board)
    EchoService.echo(str(board_array))
    y_free, x_free = get_random_free_space(board_array)
    y_pickup, x_pickup = get_random_pickup_space(board_array)
    a = 1
    # board_labelled = class_from_prediction(board_array)
