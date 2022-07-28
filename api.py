import random
import string
import time
from urllib.request import Request

from fastapi import FastAPI

import logging
from main import download_image, get_corners, slice_image, predict_chesspieces
from keras.models import load_model
from apiCheckApp.services import EchoService

app = FastAPI()
model = load_model('models/blue_red_model_200.h5', compile=False)

# setup loggers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)  # the __name__ resolve to "main" since we are at the root of the project.


# This will get the root logger since no logger in the configuration has this name.

@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")

    return response


@app.get("/")
async def root():
    logger.info("logging from the root logger")
    EchoService.echo("Starting recognition Workflow")
    image_bytes = download_image()
    corners = get_corners()
    sliced_board = slice_image(corners, image_bytes)
    board_array = predict_chesspieces(model, sliced_board)
    EchoService.echo(str(board_array))
    return str(board_array)
