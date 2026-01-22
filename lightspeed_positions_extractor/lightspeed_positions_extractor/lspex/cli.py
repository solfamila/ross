from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .config import MatchConfig, RoiConfig, RunConfig
from .timeline import process_video
from .events import compute_events
from .grouping import group_flat_to_flat

def main():
    ap = argparse.ArgumentParser(description="Extract trades from Lightspeed Positions window videos (Mode A).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--sample-fps", type=float, default=6.0)
    ap.add_argument("--diff-threshold", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.70)
    ap.add_argument("--max-seconds", type=float, default=None)

    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    match_cfg = MatchConfig(min_score=args.min_score)
    roi_cfg = RoiConfig()
    run_cfg = RunConfig(sample_fps=args.sample_fps, diff_threshold=args.diff_threshold)

    tl = process_video(args.video, args.template, str(out_dir), match_cfg, roi_cfg, run_cfg, max_seconds=args.max_seconds)
    events = compute_events(tl.snapshots)
    grouped = group_flat_to_flat(events)

    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    pd.DataFrame(grouped).to_csv(out_dir / "grouped_trades.csv", index=False)

    print("\n=== DONE ===")
    print(f"ROI: {tl.roi}  header_score={tl.header_score:.3f}")
    print(f"Samples taken: {tl.samples_taken}")
    print(f"Events: {len(events)}")
    print(f"Grouped trades: {len(grouped)}")
    print(f"Outputs written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
