import cv2
import numpy as np

video = r"C:/Users/forfr/Downloads/trade/trade1_first_10s.mp4"
frame_idx = 158

tpl_path = r"C:/Users/forfr/Downloads/trade/cpp_realtime_ocr/templates/headers/positions_hdr.png"

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, fr = cap.read()
if not ok:
    raise SystemExit("frame read failed")

gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# python-style stream rect
leftW = int(W * 0.7)
mask = (gray[:, :leftW] > 20).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
else:
    x, y, w, h = 0, 0, leftW, H

search = gray[y:y+h, x:x+int(w*0.7)]

hdr = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)

best = (-1, None, None, None)
for s in (0.8, 0.9, 1.0, 1.1):
    tw = max(8, int(hdr.shape[1]*s))
    th = max(8, int(hdr.shape[0]*s))
    if tw >= search.shape[1] or th >= search.shape[0]:
        continue
    tpl = cv2.resize(hdr, (tw, th), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx > best[0]:
        best = (mx, loc, tw, th)

score, loc, tw, th = best
hx = x + loc[0]
hy = y + loc[1]
print("stream", (x, y, w, h))
print("best", score, (hx, hy, tw, th))
