from dataclasses import dataclass

@dataclass
class MatchConfig:
    min_scale: float = 0.55
    max_scale: float = 1.55
    num_scales: int = 21
    min_score: float = 0.70
    method: int = 5  # cv2.TM_CCOEFF_NORMED

@dataclass
class RoiConfig:
    extra_bottom: int = 340
    extra_right: int = 0
    extra_left: int = 0
    extra_top: int = 0

@dataclass
class RunConfig:
    sample_fps: float = 6.0
    diff_threshold: float = 2.0
    max_rows: int = 40
    ocr_scale: int = 3
    ocr_psm: int = 7
