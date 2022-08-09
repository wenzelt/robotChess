import random

import cv2
import numpy as np
import requests
from keras.models import load_model

from apiCheckApp.services import EchoService
from utility_functions.corners import get_corners

LOCAL = False
DRAW_CORNERS = False


model = load_model("models/model_5000_blue_red.h5")


def download_image(url="https://lab.bpm.in.tum.de/img/high/url") -> bytes:
    if LOCAL:
        with open(
                "full_boards/chess_full2022-06-28 13:56:05.803477.png", "rb"
        ) as image:
            f = image.read()
        return f
    try:
        EchoService.echo("sending request")
        url_endpoint_response = requests.get(url, params={})
        img_endpoint_response = requests.get(url_endpoint_response.content, params={})
        EchoService.echo("response received")
        if img_endpoint_response.status_code == 200:
            EchoService.echo(f"status_code == {img_endpoint_response.status_code}")
            image_bytes = img_endpoint_response.content
            return image_bytes
        elif img_endpoint_response.status_code == 502:
            EchoService.echo(f"status_code == {img_endpoint_response.status_code}")
            raise SystemExit()
        else:
            raise requests.exceptions.HTTPError
    except requests.exceptions.Timeout:
        print("Timeout exception")


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

    img_cut = img[min_y - 10: max_y - 10, min_x - 10: max_x - 10]
    img_square = cv2.resize(img_cut, (square_pixel * 8, square_pixel * 8))

    field_images = [
        [
            img_square[row: row + square_pixel, column: column + square_pixel, :]
            for column in range(0, img_square.shape[1], square_pixel)
        ]
        for row in range(0, img_square.shape[0], square_pixel)
    ]

    return field_images


def save_images_to_disk() -> None:
    image = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image)
    flattened_list = [element for sublist in sliced_board for element in sublist]
    x = np.array(
        [
            np.array(cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA))
            for image in flattened_list
        ]
    )
    prediction_list = predict_chesspieces(
        model=model, field=sliced_board, reshape=(8, 8)
    )
    flattened_list_predictions = [
        element for sublist in prediction_list for element in sublist
    ]
    counter = 0
    for i in x:
        if flattened_list_predictions[counter] != 0:
            cv2.imwrite(f"images/unsorted/{counter}_{random.randint(0, 10000)}.png", i)
        counter += 1
        if counter == 63:
            break
