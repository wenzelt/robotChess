import cv2
import numpy as np
import requests
import io

from PIL import Image, ImageOps
from keras.models import load_model

from fixtures.empty_chessboard import EMPTY_CHESSBOARD
from fixtures.local_chessboard import LOCAL_CHESSBOARD

LOCAL = True
DRAW_CORNERS = False


def download_image() -> bytes:
    with open("full_boards/chess_full2022-06-28 15:24:25.338598.png", "rb") as image:
        f = image.read()

    if LOCAL:
        return f
    URL = "https://lab.bpm.in.tum.de/img/high/url"
    try:
        print("sending request")
        url_endpoint_response = requests.get(URL, params={})
        img_endpoint_response = requests.get(url_endpoint_response.content, params={})
        print("response received")
        if img_endpoint_response.status_code == 200:
            print(f"status_code == {img_endpoint_response.status_code}")
            image_bytes = img_endpoint_response.content
            return image_bytes
        elif img_endpoint_response.status_code == 502:
            print(f"status_code == {img_endpoint_response.status_code}")
            raise SystemExit()
        else:
            raise requests.exceptions.HTTPError
    except requests.exceptions.Timeout:
        print("Timeout exception")


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
        print("No Checkerboard Found")


def slice_image(corners, cam_image_in_bytes, write_to_disk: bool) -> list[list]:
    img_np = np.frombuffer(cam_image_in_bytes, dtype='uint8')
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    square_pixel = int(corners[1][1] - corners[0][1])
    ru = np.array((corners[0][0] + square_pixel, corners[0][1] - square_pixel)).astype(int)
    ld = np.array((corners[-1][0] - square_pixel, corners[-1][1] + square_pixel)).astype(int)

    min_x, max_y = ld
    max_x, min_y = ru

    img_cut = img[min_y:max_y, min_x:max_x]
    img_square = cv2.resize(img_cut, (square_pixel * 8, square_pixel * 8))

    squares = []
    alpha_desc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = [8, 7, 6, 5, 4, 3, 2, 1]
    counter_a = 0
    counter_b = 0

    field_images = [
        [{'field': f'{alpha_desc[int(column / 85)], numbers[int(row / 85)]}',
          'image': img_square[row:row + square_pixel, column:column + square_pixel, :],
          "prediction": ""} for column in
         range(0, img_square.shape[1], square_pixel)] for
        row in range(0, img_square.shape[0], square_pixel)]

    # if write_to_disk:
    #     for r in range(0, img_square.shape[0], square_pixel):
    #         for c in range(0, img_square.shape[1], square_pixel):
    #             field = f'{alpha_desc[counter_b]}{numbers[counter_a]}'
    #
    #             cv2.imwrite(f"./squares/{field}.png",
    #                         img_square[r:r + square_pixel, c:c + square_pixel, :])
    #             counter_b += 1
    #
    #             counter_b = counter_b % 8
    #         counter_a += 1
    return field_images


def class_from_prediction(prediction):
    CLASS_ARR = ["KB", "QB", "RB", "PB", "BB", "EF", "BR", "QR", 'KR', "RR", "PR"]

    predicted_class = np.argmax(prediction, axis=1)
    predicted_class = map(lambda x: CLASS_ARR[x], predicted_class)
    print(predicted_class)
    return predicted_class



def predict_chesspieces(model, field):
    flatten_list = [element for sublist in field for element in sublist]
    x = np.array([np.array(cv2.resize(image['image'], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)) for image in
                  flatten_list])

    normalized_image_array = (x.astype(np.float32) / 127.0) - 1

    prediction = model.predict(normalized_image_array)

    board_array = np.reshape(np.argmax(prediction, axis=1), (8, 8))

    return board_array


if __name__ == '__main__':
    image_bytes = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image_bytes, True)
    model = load_model('fixtures/keras_model.h5')
    board_array = predict_chesspieces(model, sliced_board)
    print(str(board_array))