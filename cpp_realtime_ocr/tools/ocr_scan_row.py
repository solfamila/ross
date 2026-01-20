import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "ross_entry_trigger_working"))
import run as r  # noqa: E402

img = cv2.imread("C:/Users/forfr/Downloads/trade/cpp_realtime_ocr/debug_best_roi.png", cv2.IMREAD_GRAYSCALE)
print("roi", None if img is None else img.shape)

best = None
best_info = None
for base in range(120, 190, 2):
    for row_h in (28, 32, 36):
        for off in (-8, -4, 0, 4, 8):
            y0 = max(0, base + off - row_h // 2)
            y1 = min(img.shape[0], y0 + row_h)
            if y1 <= y0:
                continue
            row = img[y0:y1, :]
            txt, sc = r.ocr_row_symbol(row)
            if best is None or sc > best:
                best = sc
                best_info = (base, row_h, off, y0, txt, sc)

print("best", best_info)
