import random

import cv2
import numpy as np
from keras.models import load_model

from main import get_corners, slice_image, predict_chesspieces
from utility_functions.image_downloader import download_image

model = load_model('models/model_5000_blue_red.h5')

if __name__ == '__main__':
    image = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image)
    flattened_list = [element for sublist in sliced_board for element in sublist]
    x = np.array([np.array(cv2.resize(image['image'], dsize=(224, 224), interpolation=cv2.INTER_AREA)) for image in
                  flattened_list])
    prediction_list = predict_chesspieces(model=model, field=sliced_board, reshape=(8, 8))
    flattened_list_predictions = [element for sublist in prediction_list for element in sublist]

    counter = 0
    for i in x:
        if flattened_list_predictions[counter] != 0:
            cv2.imwrite(f'images/{counter}_{random.randint(0, 1000)}.png', i)
        counter += 1
