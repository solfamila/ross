# Lightspeed Positions Video Trade Extractor (Mode A)

Extract trades from the **Lightspeed Trader "Positions" window** recorded in a video.

Mode A = per-video auto-calibration using **multi-scale template matching** of the Positions header.
- Window position/size may vary between videos.
- Window stable within a video (your case).

Outputs (in your `--out` folder):
- `timeline.jsonl`: timestamped parsed Positions snapshots
- `events.csv`: BUY/SELL/SHORT/COVER events from snapshot diffs
- `grouped_trades.csv`: trades grouped **flat → flat** (Option A)
- `debug/roi_match.png`: ROI box drawn on first frame
- `debug/roi_crop.png`: cropped ROI image for inspection
- `debug/ocr_samples/`: a few OCR sample dumps (for troubleshooting)

## Install
```bash
pip install -r requirements.txt
```

Install Tesseract:
- Windows: install Tesseract OCR and add to PATH
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`

Verify:
```bash
tesseract --version
```

## Prepare a header template
Create a PNG containing **only the Positions header strip**, ideally including:
- "Positions" and
- column headers ("Symbol", "Position size", "Cost Basis", "Open P&L", "Realized P&L")

Example: `tpl_header.png`

## Run
```bash
python -m lspex.cli \
  --video /path/to/video.mp4 \
  --template /path/to/tpl_header.png \
  --out out_run \
  --sample-fps 6 \
  --diff-threshold 2.0
```

### Quick smoke test (first 30 seconds)
```bash
python scripts/quick_test.py \
  --video /path/to/video.mp4 \
  --template /path/to/tpl_header.png
```
