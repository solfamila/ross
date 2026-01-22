from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import MatchConfig, RoiConfig, RunConfig
from .roi import find_header_roi, expand_roi, draw_roi
from .preprocess import text_mask_bgr, segment_rows
from .ocr import ocr_line, parse_row, snapshot_from_rows

@dataclass
class TimelineResult:
    roi: Tuple[int,int,int,int]
    header_score: float
    samples_taken: int
    snapshots: List[dict]

def process_video(video_path: str, template_path: str, out_dir: str,
                  match_cfg: MatchConfig, roi_cfg: RoiConfig, run_cfg: RunConfig,
                  max_seconds: Optional[float] = None) -> TimelineResult:

    outp = Path(out_dir)
    (outp / "debug" / "ocr_samples").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame")

    tpl = cv2.imread(template_path)
    if tpl is None:
        raise RuntimeError(f"Cannot read template: {template_path}")

    header = find_header_roi(first, tpl, match_cfg)
    roi = expand_roi(header, roi_cfg, first.shape)

    roi_img = draw_roi(first, roi, text=f"ROI score={header.score:.3f}")
    cv2.imwrite(str(outp / "debug" / "roi_match.png"), roi_img)
    x,y,w,h = roi
    cv2.imwrite(str(outp / "debug" / "roi_crop.png"), first[y:y+h, x:x+w])

    sample_every = max(1, int(round(fps / run_cfg.sample_fps)))

    snapshots: List[dict] = []
    prev_roi_gray = None
    samples = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing", unit="frame")
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_seconds is not None and (frame_idx / fps) > max_seconds:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        crop = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if prev_roi_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.int16) - prev_roi_gray.astype(np.int16))))
            if diff < run_cfg.diff_threshold:
                frame_idx += 1
                pbar.update(1)
                continue

        prev_roi_gray = gray
        samples += 1

        mask = text_mask_bgr(crop)

        skip_top = int(header.h + 8)
        mask_body = mask[skip_top:, :]
        crop_body = crop[skip_top:, :]

        rows_y = segment_rows(mask_body, max_rows=run_cfg.max_rows)

        parsed = {}
        sample_dump = []
        for ridx, (r0, r1) in enumerate(rows_y):
            row_img = crop_body[r0:r1, :]
            text = ocr_line(row_img, scale=run_cfg.ocr_scale, psm=run_cfg.ocr_psm)
            row = parse_row(text)
            if row:
                parsed[row.symbol] = row
            if samples <= 5 and ridx < 8:
                sample_dump.append(text)

        if samples <= 5:
            (outp / "debug" / "ocr_samples" / f"sample_{samples:03d}.txt").write_text("\n".join(sample_dump), encoding="utf-8")

        state = snapshot_from_rows(parsed)
        snapshots.append({"t": round(frame_idx / fps, 3), "state": state})

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    with open(outp / "timeline.jsonl", "w", encoding="utf-8") as f:
        for s in snapshots:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return TimelineResult(roi=roi, header_score=header.score, samples_taken=samples, snapshots=snapshots)
