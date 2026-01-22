from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
from .config import MatchConfig, RoiConfig

@dataclass
class RoiMatch:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def find_header_roi(frame_bgr: np.ndarray, template_bgr: np.ndarray, match_cfg: MatchConfig) -> RoiMatch:
    frame = _to_gray(frame_bgr)
    tpl0 = _to_gray(template_bgr)

    best: Optional[RoiMatch] = None
    scales = np.linspace(match_cfg.min_scale, match_cfg.max_scale, match_cfg.num_scales)

    for s in scales:
        tpl = cv2.resize(tpl0, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)
        th, tw = tpl.shape[:2]
        fh, fw = frame.shape[:2]
        if th >= fh or tw >= fw or th < 10 or tw < 30:
            continue

        res = cv2.matchTemplate(frame, tpl, match_cfg.method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best is None or max_val > best.score:
            best = RoiMatch(x=int(max_loc[0]), y=int(max_loc[1]), w=int(tw), h=int(th),
                            score=float(max_val), scale=float(s))

        if max_val >= match_cfg.min_score + 0.05:
            break

    if best is None or best.score < match_cfg.min_score:
        raise RuntimeError(f"Header template match failed: best_score={0.0 if best is None else best.score:.3f}. "
                           f"Try a cleaner header template or adjust --min-score.")

    return best

def expand_roi(header: RoiMatch, roi_cfg: RoiConfig, frame_shape: Tuple[int,int,int]) -> Tuple[int,int,int,int]:
    fh, fw = frame_shape[:2]
    x0 = max(0, header.x - roi_cfg.extra_left)
    y0 = max(0, header.y - roi_cfg.extra_top)
    x1 = min(fw, header.x + header.w + roi_cfg.extra_right)
    y1 = min(fh, header.y + header.h + roi_cfg.extra_bottom)
    return (x0, y0, x1 - x0, y1 - y0)

def draw_roi(frame_bgr: np.ndarray, roi: Tuple[int,int,int,int], text: str = "ROI") -> np.ndarray:
    x,y,w,h = roi
    out = frame_bgr.copy()
    cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,255), 2)
    cv2.putText(out, text, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return out
