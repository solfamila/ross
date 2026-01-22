from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import re
import os
import cv2
import pytesseract

_TESS_DEFAULT = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
_tess_cmd = os.environ.get("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
elif os.path.exists(_TESS_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESS_DEFAULT

NUM_RE = re.compile(r"[+-]?\d[\d,]*\.?\d*")

@dataclass
class PositionRow:
    symbol: str
    position: int
    cost_basis: float
    open_pnl: float
    realized_pnl: float
    raw: str

def _to_float(x: str) -> float:
    return float(x.replace(",", ""))

def _to_int(x: str) -> int:
    return int(x.replace(",", ""))

def ocr_line(img_bgr, scale: int = 3, psm: int = 7) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        gray = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,+-"
    return pytesseract.image_to_string(th, config=cfg).strip()

def parse_row(text: str) -> Optional[PositionRow]:
    text = text.replace("O", "0").replace("Q", "0").strip()
    if not text:
        return None
    parts = text.split()
    sym = parts[0].upper()
    if not re.fullmatch(r"[A-Z]{1,6}", sym):
        return None

    nums = NUM_RE.findall(text[len(parts[0]):])
    if len(nums) < 3:
        return None

    if len(nums) >= 4 and "." not in nums[0]:
        pos_s, cost_s, open_s, real_s = nums[0], nums[1], nums[2], nums[3]
    else:
        pos_s = "0"
        cost_s, open_s, real_s = nums[0], nums[1], nums[2]

    try:
        return PositionRow(
            symbol=sym,
            position=_to_int(pos_s),
            cost_basis=_to_float(cost_s),
            open_pnl=_to_float(open_s),
            realized_pnl=_to_float(real_s),
            raw=text,
        )
    except Exception:
        return None

def snapshot_from_rows(rows: Dict[str, PositionRow]) -> Dict[str, dict]:
    return {s: {"position": r.position, "cost_basis": r.cost_basis, "open_pnl": r.open_pnl,
                "realized_pnl": r.realized_pnl, "raw": r.raw} for s, r in rows.items()}
