# Ross Entry Trigger (Offline Video)

This tool detects the entry moment from an offline screen recording.

Primary method:
- Anchor the Positions panel using a header template (matchTemplate).
- Compute the **maximum per-frame change** in the symbol-column/top-rows ROI to find the first appearance / row insertion.

Optional method:
- If symbol templates are available, it also attempts symbol match.

Outputs:
- event.json (entry window, frame indices, time)
- profile.json (per-frame timings)

## Install
pip install -r requirements.txt

## Run
python run.py --video trade1_first_10s.mp4 --target APVO --out event.json --profile profile.json

Templates expected at:
templates/headers/positions_hdr.png
templates/symbols/APVO_0.png (optional)
