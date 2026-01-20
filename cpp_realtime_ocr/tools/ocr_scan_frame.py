import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "ross_entry_trigger_working"))
import run as r  # noqa: E402

video = r"C:/Users/forfr/Downloads/trade/trade1_first_10s.mp4"
frame_idx = 162

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, fr = cap.read()
if not ok:
    raise SystemExit("frame read failed")

gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

hx, hy, tw, th = 96, 396, 234, 18
# ROI under header
x = hx
y = hy + th + 5
w = 260
h = 220
H, W = gray.shape
x = max(0, min(x, W-1))
y = max(0, min(y, H-1))
w = max(1, min(w, W-x))
h = max(1, min(h, H-y))
roi = gray[y:y+h, x:x+w]

best = None
best_info = None
for base in range(120, 190, 2):
    for row_h in (28, 32, 36):
        for off in (-8, -4, 0, 4, 8):
            y0 = max(0, base + off - row_h // 2)
            y1 = min(roi.shape[0], y0 + row_h)
            if y1 <= y0:
                continue
            row = roi[y0:y1, :]
            txt, sc = r.ocr_row_symbol(row)
            if best is None or sc > best:
                best = sc
                best_info = (base, row_h, off, y0, txt, sc)

print("best", best_info)
