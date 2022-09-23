import cv2
import numpy as np

from apiCheckApp.services import EchoService
from fixtures.empty_chessboard import EMPTY_CHESSBOARD

DRAW_CORNERS = True


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
    ret, corner_coords = cv2.findChessboardCorners(
        res,
        (ncol, nline),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ret:
        if DRAW_CORNERS:
            fnl = cv2.drawChessboardCorners(img, (ncol, ncol), corner_coords, ret)
            cv2.imshow("fnl", fnl)
            cv2.imwrite("corners.png", fnl)

        return corner_coords.squeeze()

    else:
        EchoService.echo("No Checkerboard Found")
