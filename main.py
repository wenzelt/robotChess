from typing import Optional

import cv2
import numpy as np
import requests
import io

from PIL import Image, ImageOps
from keras.models import load_model

from apiCheckApp.services import EchoService
from fixtures.empty_chessboard import EMPTY_CHESSBOARD
from fixtures.local_chessboard import LOCAL_CHESSBOARD
from utility_functions.image_downloader import download_image

DRAW_CORNERS = False


def get_corners():
    ncol = 7
    nline = 7

    nparr = np.frombuffer(EMPTY_CHESSBOARD, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

    lwr = np.array([0, 0, 143])
    upr = np.array([179, 61, 252])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)

    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)

    res = np.uint8(res)
    ret, corner_coords = cv2.findChessboardCorners(res, (ncol, nline),
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_FAST_CHECK +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        if DRAW_CORNERS:
            fnl = cv2.drawChessboardCorners(img, (ncol, ncol), corner_coords, ret)
            cv2.imshow("fnl", fnl)
            cv2.imwrite('corners.png', fnl)

        return corner_coords.squeeze()

    else:
        EchoService.echo("No Checkerboard Found")


def slice_image(corners, cam_image_in_bytes) -> list[list]:
    img_np = np.frombuffer(cam_image_in_bytes, dtype='uint8')
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    square_pixel = int(corners[1][1] - corners[0][1])
    ru = np.array((corners[0][0] + square_pixel, corners[0][1] - square_pixel)).astype(int)
    ld = np.array((corners[-1][0] - square_pixel , corners[-1][1] + square_pixel)).astype(int)

    min_x, max_y = ld
    max_x, min_y = ru

    img_cut = img[min_y-10:max_y-10, min_x-10:max_x-10]
    img_square = cv2.resize(img_cut, (square_pixel * 8, square_pixel * 8))

    field_images = [
        [
            {'image': img_square[row:row + square_pixel, column:column + square_pixel, :], }
            for column in
            range(0, img_square.shape[1], square_pixel)] for
        row in range(0, img_square.shape[0], square_pixel)]

    return field_images


def class_from_prediction(board_array):
    CLASS_ARR = ["HB", "KB", "QB", "RB", "PB", 'BB', "E", "BR", "KR", 'HR', "RR", "PR"]
    predicted_class = map(lambda x: CLASS_ARR[x], list(board_array))
    return predicted_class


def predict_chesspieces(model, field):
    flattened_list = [element for sublist in field for element in sublist]
    x = np.array([np.array(cv2.resize(image['image'], dsize=(224, 224), interpolation=cv2.INTER_AREA)) for image in
                  flattened_list])

    normalized_image_array = (x.astype(np.float32) / 127.0) - 1

    prediction = model.predict(normalized_image_array)

    board_array = np.reshape(np.argmax(prediction, axis=1), (8, 8))
    return board_array


if __name__ == '__main__':
    image_bytes = download_image()
    corners = get_corners()

    sliced_board = slice_image(corners, image_bytes)
    model = load_model('models/model_5000_blue_red.h5')
    board_array = predict_chesspieces(model, sliced_board)
    EchoService.echo(str(board_array))
    # board_labelled = class_from_prediction(board_array)
