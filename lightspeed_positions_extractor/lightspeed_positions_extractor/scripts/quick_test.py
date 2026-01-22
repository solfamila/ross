import argparse
from pathlib import Path
import pandas as pd

from lspex.config import MatchConfig, RoiConfig, RunConfig
from lspex.timeline import process_video
from lspex.events import compute_events
from lspex.grouping import group_flat_to_flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="out_quick")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    match_cfg = MatchConfig(min_score=0.70)
    roi_cfg = RoiConfig()
    run_cfg = RunConfig(sample_fps=6.0, diff_threshold=2.0)

    tl = process_video(args.video, args.template, str(out), match_cfg, roi_cfg, run_cfg, max_seconds=args.seconds)
    events = compute_events(tl.snapshots)
    grouped = group_flat_to_flat(events)

    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    pd.DataFrame(grouped).to_csv(out / "grouped_trades.csv", index=False)

    print("=== QUICK TEST DONE ===")
    print(f"ROI: {tl.roi}  header_score={tl.header_score:.3f}")
    print(f"Samples taken: {tl.samples_taken}")
    print(f"Events: {len(events)}")
    print(f"Grouped trades: {len(grouped)}")
    print(f"Outputs: {out.resolve()}")

if __name__ == "__main__":
    main()
