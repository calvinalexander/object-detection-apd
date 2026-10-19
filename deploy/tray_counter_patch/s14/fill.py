import cv2, numpy as np
from det import A_SLOPE, A_ICPT
def features(img, box, mm):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inner = cv2.erode(mm, np.ones((9, 9), np.uint8)) > 0
    s, v = hsv[..., 1][inner], hsv[..., 2][inner]
    food = ((s > 45) | ((v > 200) & (s <= 45))).mean() if inner.any() else 0.0
    x1, y1, x2, y2 = box
    size = mm.sum() / (A_SLOPE * (y1 + y2) / 2 + A_ICPT)      # 1.0 = one tray's footprint
    return float(food), float(size)
