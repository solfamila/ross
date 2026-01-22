from __future__ import annotations
from typing import Dict, List, Any, Optional
import math

def infer_fill_price_add(old_pos: int, old_cost: float, new_pos: int, new_cost: float) -> Optional[float]:
    delta = new_pos - old_pos
    if delta == 0:
        return None
    return (new_cost * new_pos - old_cost * old_pos) / delta

def infer_fill_price_reduce(old_cost: float, realized_delta: float, shares: int, side: str) -> Optional[float]:
    if shares == 0:
        return None
    if side == "SELL":
        return realized_delta / shares + old_cost
    if side == "COVER":
        return old_cost - realized_delta / shares
    return None

def _get(snap: Dict[str, Any], sym: str) -> Dict[str, Any]:
    return snap.get(sym, {"position": 0, "cost_basis": 0.0, "open_pnl": 0.0, "realized_pnl": 0.0})

def compute_events(timeline: List[dict]) -> List[dict]:
    events: List[dict] = []
    for i in range(1, len(timeline)):
        prev = timeline[i-1]["state"]
        cur = timeline[i]["state"]
        t = float(timeline[i]["t"])
        for sym in sorted(set(prev.keys()) | set(cur.keys())):
            a = _get(prev, sym)
            b = _get(cur, sym)
            old_pos = int(a["position"]); new_pos = int(b["position"])
            if old_pos == new_pos:
                continue
            old_cost = float(a.get("cost_basis", 0.0))
            new_cost = float(b.get("cost_basis", 0.0))
            realized_delta = float(b.get("realized_pnl", 0.0)) - float(a.get("realized_pnl", 0.0))
            delta = new_pos - old_pos

            if delta > 0:
                if old_pos >= 0 and new_pos > old_pos:
                    side="BUY"; shares=delta
                    fill=infer_fill_price_add(old_pos, old_cost, new_pos, new_cost)
                else:
                    side="COVER"; shares=delta
                    fill=infer_fill_price_reduce(old_cost, realized_delta, shares, side)
            else:
                if old_pos > 0 and new_pos < old_pos:
                    side="SELL"; shares=-delta
                    fill=infer_fill_price_reduce(old_cost, realized_delta, shares, side)
                else:
                    side="SHORT"; shares=-delta
                    fill=infer_fill_price_add(old_pos, old_cost, new_pos, new_cost)

            if fill is not None and (math.isnan(fill) or math.isinf(fill)):
                fill = None

            events.append({
                "t": t, "symbol": sym, "side": side, "shares": int(shares),
                "pos_before": old_pos, "pos_after": new_pos,
                "old_cost": round(old_cost,4), "new_cost": round(new_cost,4),
                "realized_delta": round(realized_delta,2),
                "inferred_fill_price": None if fill is None else round(float(fill),4),
            })
    return events
