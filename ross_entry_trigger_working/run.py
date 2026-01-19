
import argparse, os, json, time, glob
import cv2
import numpy as np

def now_ms(): return time.perf_counter()*1000.0

def clamp_rect(x,y,w,h,W,H):
    x=max(0,min(int(x),W-1)); y=max(0,min(int(y),H-1))
    w=max(1,min(int(w),W-x)); h=max(1,min(int(h),H-y))
    return x,y,w,h

def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def match_best(gray, tpl):
    res=cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _,mx,_,loc=cv2.minMaxLoc(res)
    return float(mx), (int(loc[0]),int(loc[1]))

def build_glyph_templates(size, font_scale, thickness):
    w,h = size
    glyphs = {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        img = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(img, ch, (1, h-6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        glyphs[ch] = img
    return glyphs

def ocr_row_symbol(gray_row):
    if gray_row is None or gray_row.size == 0:
        return "", -1.0
    # focus on left side where symbol appears
    gray_row = gray_row[:, :min(220, gray_row.shape[1])]
    # upscale for better tiny OCR
    scale = 2
    gray_row = cv2.resize(gray_row, (gray_row.shape[1]*scale, gray_row.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    # binarize
    _, bw = cv2.threshold(gray_row, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 40:
            continue
        if h < 12 or h > 80 or w < 5 or w > 60:
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return "", -1.0
    # take first 6 characters
    boxes = boxes[:6]
    # build glyphs based on row height
    row_h = max(18, int(gray_row.shape[0] * 0.9))
    glyph_w = max(12, int(row_h * 0.6))
    font_scale = 0.6 * (row_h / 30.0)
    thickness = 1 if row_h < 28 else 2
    glyphs = build_glyph_templates((glyph_w, row_h), font_scale, thickness)
    chars = []
    scores = []
    for (x,y,w,h) in boxes:
        patch = bw[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        best_ch = "?"
        best_sc = -1.0
        for ch, tpl in glyphs.items():
            ph, pw = tpl.shape[:2]
            patch_r = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(patch_r, tpl, cv2.TM_CCORR_NORMED)
            sc = float(cv2.minMaxLoc(res)[1])
            if sc > best_sc:
                best_sc = sc
                best_ch = ch
        chars.append(best_ch)
        scores.append(best_sc)
    text = "".join(chars)
    score = float(np.mean(scores)) if scores else -1.0
    return text, score

def match_word(gray_row, word):
    if gray_row is None or gray_row.size == 0:
        return -1.0
    gray_row = gray_row[:, :min(260, gray_row.shape[1])]
    scale = 2
    gray_row = cv2.resize(gray_row, (gray_row.shape[1]*scale, gray_row.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    best = -1.0
    for s in (0.8, 1.0, 1.2):
        h = max(20, int(gray_row.shape[0] * 0.9))
        w = max(40, int(h * 0.7 * len(word)))
        tpl = np.zeros((h, w), dtype=np.uint8)
        font_scale = 0.6 * (h / 30.0) * s
        thickness = 1 if h < 28 else 2
        cv2.putText(tpl, word, (2, h-6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        if tpl.shape[0] >= gray_row.shape[0] or tpl.shape[1] >= gray_row.shape[1]:
            continue
        sc, _ = match_best(gray_row, tpl)
        best = max(best, sc)
    return best

def find_stream_rect(gray):
    H,W=gray.shape
    leftW=int(W*0.7)
    mask=(gray[:, :leftW]>20).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
    cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0,0,leftW,H)
    cnt=max(cnts, key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(cnt)
    return (x,y,w,h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="event.json")
    ap.add_argument("--profile", default="profile.json")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--header-thresh", type=float, default=0.55)
    ap.add_argument("--symbol-thresh", type=float, default=0.60)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--mode", choices=["symbol", "change"], default="symbol")
    ap.add_argument("--confirm-frames", type=int, default=2)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    ap.add_argument("--track-header", action="store_true")
    args=ap.parse_args()

    ACQUIRE_SCALES = [0.80, 0.90, 1.00, 1.10]
    TRACK_SCALES_MULT = [0.97, 1.00, 1.03]
    TRACK_SR = 60
    TRACK_MAX_SHIFT = 25
    TRACK_REACQUIRE_EVERY = 15
    TRACK_MAX_BASE_SHIFT_X = 80
    TRACK_MAX_BASE_SHIFT_Y = 50
    use_track = args.track_header

    header_path=os.path.join(args.templates,"headers","positions_hdr.png")
    if not os.path.exists(header_path):
        raise SystemExit(f"Missing header template: {header_path}")
    hdr_tpl=cv2.imread(header_path, cv2.IMREAD_GRAYSCALE)
    if hdr_tpl is None:
        raise SystemExit("Failed to load header template")

    sym_paths=sorted(glob.glob(os.path.join(args.templates,"symbols",f"{args.target}_*.png")))
    sym_tpls=[cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sym_paths]
    sym_tpls=[t for t in sym_tpls if t is not None]

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Failed to open video")

    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ok, fr0 = cap.read()
    if not ok: raise SystemExit("Failed to read first frame")
    g0=to_gray(fr0)
    H,W=g0.shape
    rx,ry,rw,rh = find_stream_rect(g0)
    search_x,search_y,search_w,search_h = rx,ry,int(rw*0.70),rh

    start_frame = max(0, args.start_frame)
    end_frame = args.end_frame if args.end_frame >= 0 else n - 1
    end_frame = min(end_frame, n - 1)

    # Rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_roi=None
    best_change=-1.0
    best_change_frame=0
    best_change_roi=None
    best_change_row_y=None

    best_sym=-1.0
    best_sym_frame=None
    best_sym_full=-1.0
    best_sym_full_frame=None

    present_count = 0
    last_absent_frame = 0
    present_first_frame = None
    symbol_event = None

    have_track = False
    track_hx = 0
    track_hy = 0
    track_tw = 0
    track_th = 0
    track_scale = 1.0
    track_fail_count = 0
    track_thresh = max(0.40, args.header_thresh - 0.05)
    track_age = 0
    track_base_hx = 0
    track_base_hy = 0

    prof=[]
    frame_idx=start_frame

    while True:
        if args.step>1 and (frame_idx % args.step)!=0:
            cap.grab()
            frame_idx += 1
            if frame_idx > end_frame: break
            continue

        t0=now_ms()
        ok, fr=cap.read()
        if not ok:
            break
        t1=now_ms()
        g=to_gray(fr)

        # header match in search region (acquire -> track -> reacquire)
        search=g[search_y:search_y+search_h, search_x:search_x+search_w]
        h0=now_ms()
        best_tpl = None
        tpl_h = hdr_tpl.shape[0]
        hscore = -1.0

        if use_track and have_track:
            track_age += 1
            if track_age >= TRACK_REACQUIRE_EVERY:
                have_track = False
                track_age = 0

        if use_track and not have_track:
            # acquire (multi-scale)
            best_h = -1.0
            best_loc = (0, 0)
            best_tpl = None
            best_s = 1.0
            for s in ACQUIRE_SCALES:
                tw = max(8, int(hdr_tpl.shape[1] * s))
                th = max(8, int(hdr_tpl.shape[0] * s))
                if tw >= search.shape[1] or th >= search.shape[0]:
                    continue
                tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                sc, loc = match_best(search, tpl_s)
                if sc > best_h:
                    best_h = sc
                    best_loc = loc
                    best_tpl = tpl_s
                    best_s = s
            hscore = best_h
            hx, hy = best_loc

            if hscore >= args.header_thresh and best_tpl is not None:
                have_track = True
                track_scale = best_s
                track_tw, track_th = best_tpl.shape[1], best_tpl.shape[0]
                track_hx = search_x + hx
                track_hy = search_y + hy
                track_base_hx = track_hx
                track_base_hy = track_hy
                track_fail_count = 0
                track_age = 0
                tpl_h = track_th
        elif use_track:
            # track in local window
            sx = max(search_x, track_hx - TRACK_SR)
            sy = max(search_y, track_hy - TRACK_SR)
            ex = min(search_x + search_w, track_hx + track_tw + TRACK_SR)
            ey = min(search_y + search_h, track_hy + track_th + TRACK_SR)
            sw = ex - sx
            sh = ey - sy
            local = g[sy:sy+sh, sx:sx+sw]

            best_h = -1.0
            best_loc = (0, 0)
            best_tpl = None
            best_s = track_scale
            for mult in TRACK_SCALES_MULT:
                s = max(0.60, min(1.40, track_scale * mult))
                tw = max(8, int(hdr_tpl.shape[1] * s))
                th = max(8, int(hdr_tpl.shape[0] * s))
                if tw >= local.shape[1] or th >= local.shape[0]:
                    continue
                tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                sc, loc = match_best(local, tpl_s)
                if sc > best_h:
                    best_h = sc
                    best_loc = loc
                    best_tpl = tpl_s
                    best_s = s

            hscore = best_h
            if hscore >= track_thresh and best_tpl is not None:
                dx = abs((sx + best_loc[0]) - track_hx)
                dy = abs((sy + best_loc[1]) - track_hy)
                if dx > TRACK_MAX_SHIFT or dy > TRACK_MAX_SHIFT:
                    track_fail_count += 1
                    prev_roi = None
                    if track_fail_count >= 3:
                        have_track = False
                    # fall through to acquire on this frame
                cand_hx = sx + best_loc[0]
                cand_hy = sy + best_loc[1]
                if abs(cand_hx - track_base_hx) > TRACK_MAX_BASE_SHIFT_X or abs(cand_hy - track_base_hy) > TRACK_MAX_BASE_SHIFT_Y:
                    track_fail_count += 1
                    prev_roi = None
                    if track_fail_count >= 3:
                        have_track = False
                    # fall through to acquire on this frame
                track_scale = best_s
                track_tw, track_th = best_tpl.shape[1], best_tpl.shape[0]
                track_hx = cand_hx
                track_hy = cand_hy
                track_fail_count = 0
                track_age = 0

                hx = track_hx - search_x
                hy = track_hy - search_y
                tpl_h = track_th
            else:
                track_fail_count += 1
                prev_roi = None
                have_track = False

            if not have_track:
                # acquire on this same frame
                best_h = -1.0
                best_loc = (0, 0)
                best_tpl = None
                best_s = 1.0
                for s in ACQUIRE_SCALES:
                    tw = max(8, int(hdr_tpl.shape[1] * s))
                    th = max(8, int(hdr_tpl.shape[0] * s))
                    if tw >= search.shape[1] or th >= search.shape[0]:
                        continue
                    tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                    sc, loc = match_best(search, tpl_s)
                    if sc > best_h:
                        best_h = sc
                        best_loc = loc
                        best_tpl = tpl_s
                        best_s = s
                hscore = best_h
                hx, hy = best_loc
                if hscore >= args.header_thresh and best_tpl is not None:
                    have_track = True
                    track_scale = best_s
                    track_tw, track_th = best_tpl.shape[1], best_tpl.shape[0]
                    track_hx = search_x + hx
                    track_hy = search_y + hy
                    track_base_hx = track_hx
                    track_base_hy = track_hy
                    track_fail_count = 0
                    track_age = 0
                    tpl_h = track_th
        else:
            # full multi-scale each frame (offline)
            best_h = -1.0
            best_loc = (0, 0)
            best_tpl = None
            for s in ACQUIRE_SCALES:
                tw = max(8, int(hdr_tpl.shape[1] * s))
                th = max(8, int(hdr_tpl.shape[0] * s))
                if tw >= search.shape[1] or th >= search.shape[0]:
                    continue
                tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                sc, loc = match_best(search, tpl_s)
                if sc > best_h:
                    best_h = sc
                    best_loc = loc
                    best_tpl = tpl_s
            hscore = best_h
            hx, hy = best_loc
            tpl_h = best_tpl.shape[0] if best_tpl is not None else hdr_tpl.shape[0]

        h1=now_ms()

        frame_best_sym = -1.0
        if hscore >= args.header_thresh:
            hx = search_x + hx
            hy = search_y + hy
            # ROI under header: left side, top rows
            tx = hx
            # tpl_h already computed in acquire/track block
            ty = hy + tpl_h + 5
            tw = 260
            th = 220
            tx,ty,tw,th = clamp_rect(tx,ty,tw,th,W,H)
            roi = g[ty:ty+th, tx:tx+tw]

            # change score
            c0=now_ms()
            if prev_roi is not None and roi.size and prev_roi.shape==roi.shape:
                diff = cv2.absdiff(roi, prev_roi)
                ch=float(diff.mean())
                if ch > best_change:
                    best_change=ch
                    best_change_frame=frame_idx
                    row_scores = diff.mean(axis=1) if diff.ndim == 2 else diff.mean(axis=2).mean(axis=1)
                    row_y = int(np.argmax(row_scores)) if len(row_scores) else 0
                    best_change_row_y = row_y
                    best_change_roi = roi.copy()
            prev_roi = roi.copy()
            c1=now_ms()

            # optional symbol match
            m0=now_ms()
            if sym_tpls and roi.size:
                for st in sym_tpls:
                    if st.shape[0] < roi.shape[0] and st.shape[1] < roi.shape[1]:
                        rr=cv2.matchTemplate(roi, st, cv2.TM_CCOEFF_NORMED)
                        _,mx,_,_=cv2.minMaxLoc(rr)
                        frame_best_sym = max(frame_best_sym, float(mx))
                if frame_best_sym > best_sym:
                    best_sym = frame_best_sym
                    best_sym_frame = frame_idx
            m1=now_ms()

            if args.mode == "symbol" and frame_best_sym >= 0:
                present = frame_best_sym >= args.symbol_thresh
                if present:
                    present_count += 1
                    if present_count == 1:
                        present_first_frame = frame_idx
                    if present_count >= args.confirm_frames and symbol_event is None:
                        absent = last_absent_frame
                        present_f = present_first_frame if present_first_frame is not None else frame_idx
                        symbol_event = {
                            "target": args.target,
                            "method": "symbol_gate",
                            "absent_last_frame": absent,
                            "present_first_frame": present_f,
                            "absent_last_time_ms": 1000.0*absent/fps,
                            "present_first_time_ms": 1000.0*present_f/fps,
                            "window_ms": [1000.0*absent/fps, 1000.0*present_f/fps],
                            "best_change": best_change,
                            "best_symbol_score": frame_best_sym,
                            "best_symbol_frame": frame_idx
                        }
                        break
                else:
                    present_count = 0
                    last_absent_frame = frame_idx
        else:
            prev_roi = None  # reset diff state when header isn't found
            present_count = 0
            last_absent_frame = frame_idx
            c0=c1=m0=m1=now_ms()

        # Full-frame symbol match (left half) only when header is missing
        if sym_tpls and hscore < args.header_thresh:
            left = g[:, :g.shape[1]//2]
            for st in sym_tpls:
                if st is None:
                    continue
                if st.shape[0] >= left.shape[0] or st.shape[1] >= left.shape[1]:
                    continue
                rr = cv2.matchTemplate(left, st, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(rr)
                if float(mx) > best_sym_full:
                    best_sym_full = float(mx)
                    best_sym_full_frame = frame_idx

        prof.append({
            "frame": frame_idx,
            "t_ms": 1000.0*frame_idx/fps,
            "decode_ms": t1-t0,
            "header_ms": h1-h0,
            "change_ms": c1-c0,
            "match_ms": m1-m0,
            "hscore": hscore,
            "symbol_score": frame_best_sym
        })

        frame_idx += 1
        if frame_idx > end_frame: break

    cap.release()

    # If tracking is enabled, stabilize best symbol frame using full-frame search in the window
    if use_track and sym_tpls:
        cap_sf = cv2.VideoCapture(args.video)
        best_full = -1.0
        best_full_frame = None
        for f in range(start_frame, end_frame + 1):
            cap_sf.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap_sf.read()
            if not ok:
                continue
            g = to_gray(fr)
            left = g[:, :g.shape[1]//2]
            for st in sym_tpls:
                if st is None:
                    continue
                if st.shape[0] >= left.shape[0] or st.shape[1] >= left.shape[1]:
                    continue
                rr = cv2.matchTemplate(left, st, cv2.TM_CCOEFF_NORMED)
                _, mx, _, _ = cv2.minMaxLoc(rr)
                if float(mx) > best_full:
                    best_full = float(mx)
                    best_full_frame = f
        cap_sf.release()
        if best_full_frame is not None:
            best_sym_frame = best_full_frame
            best_sym_full_frame = best_full_frame

    # Locate target row near best symbol frame using word match
    target_row_y = None
    target_row_h = 32
    target_band = None  # (x, y, w, h) in full frame
    if best_sym_frame is not None:
        cap2 = cv2.VideoCapture(args.video)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, best_sym_frame)
        ok, fr = cap2.read()
        cap2.release()
        if ok:
            g = to_gray(fr)
            search=g[search_y:search_y+search_h, search_x:search_x+search_w]
            scales = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
            best_h = -1.0
            best_loc = (0, 0)
            best_tpl = None
            for s in scales:
                tw = max(8, int(hdr_tpl.shape[1] * s))
                th = max(8, int(hdr_tpl.shape[0] * s))
                if tw >= search.shape[1] or th >= search.shape[0]:
                    continue
                tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                sc, loc = match_best(search, tpl_s)
                if sc > best_h:
                    best_h = sc
                    best_loc = loc
                    best_tpl = tpl_s
            # Prefer direct symbol localization for row band
            if sym_tpls:
                best_sym_loc = None
                best_sym_sc = -1.0
                left = g[:, :g.shape[1]//2]
                for st in sym_tpls:
                    if st is None:
                        continue
                    if st.shape[0] >= left.shape[0] or st.shape[1] >= left.shape[1]:
                        continue
                    rr = cv2.matchTemplate(left, st, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(rr)
                    if float(mx) > best_sym_sc:
                        best_sym_sc = float(mx)
                        best_sym_loc = loc
                if best_sym_loc is not None:
                    x = max(0, best_sym_loc[0] - 10)
                    y = max(0, best_sym_loc[1] - 4)
                    w = min(260, g.shape[1] - x)
                    h = min(40, g.shape[0] - y)
                    target_band = (x, y, w, h)
                    target_row_y = 0
                    target_row_h = h

            if best_h >= args.header_thresh and target_band is None:
                hx = search_x + best_loc[0]
                hy = search_y + best_loc[1]
                tpl_h = best_tpl.shape[0] if best_tpl is not None else hdr_tpl.shape[0]
                tx = hx
                ty = hy + tpl_h + 5
                tw = 260
                th = 220
                tx,ty,tw,th = clamp_rect(tx,ty,tw,th,W,H)
                roi = g[ty:ty+th, tx:tx+tw]
                # Prefer template match to locate the symbol row, then fallback to word match
                best_row_score = -1.0
                if sym_tpls:
                    for st in sym_tpls:
                        if st is None:
                            continue
                        if st.shape[0] >= roi.shape[0] or st.shape[1] >= roi.shape[1]:
                            continue
                        rr = cv2.matchTemplate(roi, st, cv2.TM_CCOEFF_NORMED)
                        _, mx, _, loc = cv2.minMaxLoc(rr)
                        if float(mx) > best_row_score:
                            best_row_score = float(mx)
                            target_row_y = int(loc[1])
                            target_row_h = max(target_row_h, st.shape[0] + 8)
                            target_band = (tx, ty + target_row_y, tw, min(target_row_h, roi.shape[0] - target_row_y))
                if target_row_y is None:
                    for y in range(0, max(1, roi.shape[0]-target_row_h), 4):
                        row_patch = roi[y:y+target_row_h, :]
                        score = match_word(row_patch, args.target)
                        if score > best_row_score:
                            best_row_score = score
                            target_row_y = y
                            target_band = (tx, ty + target_row_y, tw, min(target_row_h, roi.shape[0] - target_row_y))

    # If we found a symbol peak, refine change-peak in a window around it
    if best_sym_full_frame is not None:
        best_sym_frame = best_sym_full_frame
        win = 30
        start = max(0, best_sym_frame - win)
        end = min(n-1, best_sym_frame + win)
        cap2 = cv2.VideoCapture(args.video)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
        prev_roi = None
        prev_gray = None
        best_change_w = -1.0
        best_frame_w = best_sym_frame
        best_roi_w = None
        best_row_w = None
        for frame_idx in range(start, end+1):
            ok, fr = cap2.read()
            if not ok:
                break
            g = to_gray(fr)
            search=g[search_y:search_y+search_h, search_x:search_x+search_w]
            # multi-scale header match
            scales = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
            best_h = -1.0
            best_loc = (0, 0)
            best_tpl = None
            for s in scales:
                tw = max(8, int(hdr_tpl.shape[1] * s))
                th = max(8, int(hdr_tpl.shape[0] * s))
                if tw >= search.shape[1] or th >= search.shape[0]:
                    continue
                tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                sc, loc = match_best(search, tpl_s)
                if sc > best_h:
                    best_h = sc
                    best_loc = loc
                    best_tpl = tpl_s
            roi = None
            if best_h >= args.header_thresh:
                hx = search_x + best_loc[0]
                hy = search_y + best_loc[1]
                tpl_h = best_tpl.shape[0] if best_tpl is not None else hdr_tpl.shape[0]
                tx = hx
                ty = hy + tpl_h + 5
                tw = 260
                th = 220
                tx,ty,tw,th = clamp_rect(tx,ty,tw,th,W,H)
                roi = g[ty:ty+th, tx:tx+tw]

            if target_band is not None:
                # compute change on fixed band in full frame
                if prev_gray is not None:
                    x,y,w,h = target_band
                    band = cv2.absdiff(g[y:y+h, x:x+w], prev_gray[y:y+h, x:x+w])
                    ch = float(band.mean()) if band.size else -1.0
                    if ch > best_change_w:
                        best_change_w = ch
                        best_frame_w = frame_idx
                        best_row_w = 0
                        best_roi_w = g[y:y+h, x:x+w].copy()
                prev_gray = g.copy()
            elif roi is not None:
                if prev_roi is not None and roi.size and prev_roi.shape==roi.shape:
                    diff = cv2.absdiff(roi, prev_roi)
                    if target_row_y is not None:
                        y0 = max(0, target_row_y)
                        y1 = min(roi.shape[0], y0 + target_row_h)
                        band = diff[y0:y1, :]
                        ch = float(band.mean()) if band.size else float(diff.mean())
                        row_y = target_row_y
                    else:
                        ch = float(diff.mean())
                        row_scores = diff.mean(axis=1) if diff.ndim == 2 else diff.mean(axis=2).mean(axis=1)
                        row_y = int(np.argmax(row_scores)) if len(row_scores) else 0
                    if ch > best_change_w:
                        best_change_w = ch
                        best_frame_w = frame_idx
                        best_row_w = row_y
                        best_roi_w = roi.copy()
                prev_roi = roi.copy() if roi is not None else None
        cap2.release()
        if best_change_w > 0:
            best_change = best_change_w
            best_change_frame = best_frame_w
            best_change_roi = best_roi_w
            best_change_row_y = best_row_w

    if best_change < 0:
        raise SystemExit("Header template never matched. You need a header template from THIS mp4 or lower --header-thresh.")

    if symbol_event is not None and args.mode == "symbol":
        event = symbol_event
    else:
        if best_change < 0:
            raise SystemExit("Header template never matched. You need a header template from THIS mp4 or lower --header-thresh.")
        # OCR confirm on change-peak row
        ocr_text = ""
        ocr_score = -1.0
        if best_change_roi is not None:
            # Prefer OCR on the frame after the change (row is present)
            ocr_roi = best_change_roi
            if best_change_frame + 1 < n:
                cap3 = cv2.VideoCapture(args.video)
                cap3.set(cv2.CAP_PROP_POS_FRAMES, best_change_frame + 1)
                ok, fr = cap3.read()
                cap3.release()
                if ok:
                    g = to_gray(fr)
                    search=g[search_y:search_y+search_h, search_x:search_x+search_w]
                    scales = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
                    best_h = -1.0
                    best_loc = (0, 0)
                    best_tpl = None
                    for s in scales:
                        tw = max(8, int(hdr_tpl.shape[1] * s))
                        th = max(8, int(hdr_tpl.shape[0] * s))
                        if tw >= search.shape[1] or th >= search.shape[0]:
                            continue
                        tpl_s = cv2.resize(hdr_tpl, (tw, th), interpolation=cv2.INTER_AREA)
                        sc, loc = match_best(search, tpl_s)
                        if sc > best_h:
                            best_h = sc
                            best_loc = loc
                            best_tpl = tpl_s
                    if best_h >= args.header_thresh:
                        hx = search_x + best_loc[0]
                        hy = search_y + best_loc[1]
                        tpl_h = best_tpl.shape[0] if best_tpl is not None else hdr_tpl.shape[0]
                        tx = hx
                        ty = hy + tpl_h + 5
                        tw = 260
                        th = 220
                        tx,ty,tw,th = clamp_rect(tx,ty,tw,th,W,H)
                        ocr_roi = g[ty:ty+th, tx:tx+tw]

            base_y = best_change_row_y if best_change_row_y is not None else (target_row_y or 0)
            best_ocr = ("", -1.0)
            best_row_patch = None
            for row_h in (28, 32, 36):
                for off in (-8, -4, 0, 4, 8):
                    y0 = max(0, base_y + off - row_h//2)
                    y1 = min(ocr_roi.shape[0], y0 + row_h)
                    row_patch = ocr_roi[y0:y1, :]
                    text, score = ocr_row_symbol(row_patch)
                    if score > best_ocr[1]:
                        best_ocr = (text, score)
                        best_row_patch = row_patch
            ocr_text, ocr_score = best_ocr
        if not ocr_text or not ocr_text.startswith(args.target[:len(ocr_text)]):
            word_score = -1.0
            word_patch = None
            tpl_score = -1.0
            if best_change_roi is not None:
                base_y = best_change_row_y or 0
                y0 = max(0, base_y - 14)
                y1 = min(best_change_roi.shape[0], y0 + 28)
                word_patch = best_change_roi[y0:y1, :]
                word_score = match_word(word_patch, args.target)
                if sym_tpls and word_patch is not None and word_patch.size:
                    for st in sym_tpls:
                        if st is None:
                            continue
                        if st.shape[0] >= word_patch.shape[0] or st.shape[1] >= word_patch.shape[1]:
                            continue
                        rr = cv2.matchTemplate(word_patch, st, cv2.TM_CCOEFF_NORMED)
                        _, mx, _, _ = cv2.minMaxLoc(rr)
                        tpl_score = max(tpl_score, float(mx))
            if word_score >= 0.50:
                ocr_text = args.target
                ocr_score = word_score
            elif tpl_score >= 0.55:
                ocr_text = args.target
                ocr_score = tpl_score
            else:
                # Debug dumps
                if best_row_patch is not None:
                    cv2.imwrite("debug_row.png", best_row_patch)
                    _, dbg_bw = cv2.threshold(best_row_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cv2.imwrite("debug_row_bw.png", dbg_bw)
                if word_patch is not None:
                    cv2.imwrite("debug_word_patch.png", word_patch)
                raise SystemExit(f"OCR mismatch at change peak. Got '{ocr_text}' (score {ocr_score:.3f}), word score {word_score:.3f}, tpl score {tpl_score:.3f}.")
        # Build event window from best_change_frame (approx): use previous frame as absent, this as present.
        absent = max(0, best_change_frame - 1)
        present = best_change_frame
        event = {
            "target": args.target,
            "method": "change_peak_ocr",
            "absent_last_frame": absent,
            "present_first_frame": present,
            "absent_last_time_ms": 1000.0*absent/fps,
            "present_first_time_ms": 1000.0*present/fps,
            "window_ms": [1000.0*absent/fps, 1000.0*present/fps],
            "best_change": best_change,
            "best_symbol_score": best_sym,
            "best_symbol_frame": best_sym_frame,
            "ocr_text": ocr_text,
            "ocr_score": ocr_score
        }

    out = {"video": args.video, "fps": fps, "event": event}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(args.profile, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)

    print(json.dumps(event, indent=2))

if __name__ == "__main__":
    main()
