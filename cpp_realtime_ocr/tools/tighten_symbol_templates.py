import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def find_foreground_bbox(gray: np.ndarray, pad: int = 2):
    h, w = gray.shape[:2]
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg = np.median(border)
    mask = np.abs(gray.astype(np.int16) - int(bg)) > 10
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return 0, 0, w, h
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Template directory")
    ap.add_argument("--prefix", default="APVO_", help="Filename prefix to tighten")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tdir = Path(args.dir)
    if not tdir.is_dir():
        raise SystemExit(f"Missing template dir: {tdir}")

    paths = sorted(p for p in tdir.iterdir() if p.name.startswith(args.prefix) and p.suffix.lower() == ".png")
    if not paths:
        raise SystemExit("No templates matched.")

    for p in paths:
        if "region" in p.stem.lower():
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Skip (failed to read): {p.name}")
            continue
        x, y, w, h = find_foreground_bbox(gray)
        if w <= 0 or h <= 0:
            print(f"Skip (empty mask): {p.name}")
            continue
        crop = gray[y:y+h, x:x+w]
        print(f"{p.name}: {gray.shape[1]}x{gray.shape[0]} -> {w}x{h} @ ({x},{y})")
        if args.dry_run:
            continue
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            os.replace(p, bak)
        cv2.imwrite(str(p), crop)


if __name__ == "__main__":
    main()
