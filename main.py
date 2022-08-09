from keras.models import load_model

from apiCheckApp.services import EchoService
from utility_functions.board_functions import predict_chesspieces, get_random_free_space, get_random_pickup_space
from utility_functions.corners import get_corners
from utility_functions.image_utils import download_image, slice_image

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
