import base64

import cv2
import numpy as np

from fixtures.empty_chessboard import EMPTY_CHESSBOARD

if __name__ == '__main__':
    string = EMPTY_CHESSBOARD
    jpg_original = base64.b64decode(string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    cv2.imwrite('./0.jpg', img)
