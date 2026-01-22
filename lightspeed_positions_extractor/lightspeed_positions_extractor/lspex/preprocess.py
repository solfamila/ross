from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2

def text_mask_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, 190), (180, 40, 255))
    green = cv2.inRange(hsv, (35, 50, 70), (90, 255, 255))
    red1 = cv2.inRange(hsv, (0, 50, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 50, 70), (180, 255, 255))

    mask = cv2.bitwise_or(white, green)
    mask = cv2.bitwise_or(mask, red1)
    mask = cv2.bitwise_or(mask, red2)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def segment_rows(mask: np.ndarray, min_row_h: int = 10, min_ink: int = 15, max_rows: int = 40) -> List[Tuple[int,int]]:
    proj = np.sum(mask > 0, axis=1)
    rows: List[Tuple[int,int]] = []
    active = False
    start = 0
    for i, v in enumerate(proj):
        if v >= min_ink and not active:
            active = True
            start = i
        elif v < min_ink and active:
            end = i
            if end - start >= min_row_h:
                rows.append((start, end))
            active = False
    if active:
        end = mask.shape[0]
        if end - start >= min_row_h:
            rows.append((start, end))
    return rows[:max_rows]
