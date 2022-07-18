from fastapi import FastAPI

from main import download_image, get_corners, slice_image, predict_chesspieces
from keras.models import load_model

app = FastAPI()
model = load_model('models/keras_model.h5', compile=False)


@app.get("/")
async def root():
    image_bytes = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image_bytes)
    board_array = predict_chesspieces(model, sliced_board)
    return str(board_array)
