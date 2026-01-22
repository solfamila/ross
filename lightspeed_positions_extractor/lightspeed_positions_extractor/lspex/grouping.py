from __future__ import annotations
from typing import List, Dict, Any

def group_flat_to_flat(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        by_sym.setdefault(e["symbol"], []).append(e)

    for sym, evs in by_sym.items():
        evs = sorted(evs, key=lambda x: x["t"])
        cur = None
        for e in evs:
            if cur is None:
                cur = {
                    "symbol": sym,
                    "entry_time": e["t"],
                    "exit_time": None,
                    "direction": "LONG" if e["pos_after"] > 0 else "SHORT" if e["pos_after"] < 0 else "UNKNOWN",
                    "max_abs_pos": abs(int(e["pos_after"])),
                    "num_adds": 0,
                    "num_partials": 0,
                    "total_realized": 0.0,
                    "events": 0,
                }

            cur["events"] += 1
            cur["total_realized"] += float(e.get("realized_delta", 0.0))
            cur["max_abs_pos"] = max(cur["max_abs_pos"], abs(int(e["pos_after"])))

            if abs(int(e["pos_after"])) > abs(int(e["pos_before"])):
                cur["num_adds"] += 1
            else:
                cur["num_partials"] += 1

            if int(e["pos_after"]) == 0:
                cur["exit_time"] = e["t"]
                cur["hold_seconds"] = round(float(cur["exit_time"]) - float(cur["entry_time"]), 3)
                cur["total_realized"] = round(cur["total_realized"], 2)
                grouped.append(cur)
                cur = None

        if cur is not None:
            cur["exit_time"] = None
            cur["hold_seconds"] = None
            cur["total_realized"] = round(cur["total_realized"], 2)
            grouped.append(cur)

    return grouped
