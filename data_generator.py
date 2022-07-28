import random

import cv2
import numpy as np

from main import get_corners, slice_image
from utility_functions.image_downloader import download_image

if __name__ == '__main__':
    image = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image)
    flattened_list = [element for sublist in sliced_board for element in sublist]
    x = np.array([np.array(cv2.resize(image['image'], dsize=(224, 224), interpolation=cv2.INTER_AREA)) for image in
                  flattened_list])

    counter = 0
    for i in x:
        counter += 1
        if counter < 17 or counter > 48:
            cv2.imwrite(f'images/{counter}_{random.randint(0, 1000)}.png', i)


