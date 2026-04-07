# Cell 2: Provide input video
from pathlib import Path
import importlib.util

# Preferred filename for your raw capture
PREFERRED_INPUT_VIDEO = "test_video.mp4"

# Primary candidates (raw videos)
VIDEO_CANDIDATES = [
    PREFERRED_INPUT_VIDEO,
    "input.mp4",
    "sample.mp4",
]

# Colab upload flow (optional)
uploaded = {}
try:
    colab_spec = importlib.util.find_spec("google.colab")
except ModuleNotFoundError:
    colab_spec = None

if colab_spec is not None:
    colab_module = __import__("google.colab", fromlist=["files"])
    files = getattr(colab_module, "files", None)
    if files is not None:
        print(f"Upload '{PREFERRED_INPUT_VIDEO}' if it is not already in the working directory.")
        uploaded = files.upload()
else:
    print("Colab uploader not available. Using local file system mode.")

# Auto-discover local raw mp4 files, excluding generated outputs
for p in sorted(Path(".").glob("*.mp4")):
    if p.name.startswith("output_"):
        continue
    if p.name not in VIDEO_CANDIDATES:
        VIDEO_CANDIDATES.append(p.name)

# Select input path
INPUT_VIDEO = None
for name in VIDEO_CANDIDATES:
    if Path(name).exists() or (name in uploaded):
        INPUT_VIDEO = name
        break

# Last-resort fallback to generated output (not recommended)
if INPUT_VIDEO is None and Path("output_colab.mp4").exists():
    INPUT_VIDEO = "output_colab.mp4"
    print("Warning: using output_colab.mp4 as input fallback. This may reduce counting quality.")

if INPUT_VIDEO:
    print(f"Input video ready: {INPUT_VIDEO}")
else:
    print("Input video not found.")
    print(f"Checked: {VIDEO_CANDIDATES}")
    print("Place your raw video in the notebook working directory, then rerun this cell.")

# Cell 3 – BoT-SORT Plant Counter with Fixes 1-5
#
# Fix 1: Event audit CSV + missed-track log (diagnostic foundation)
# Fix 2: NMS IoU → 0.35 (prevent dense-row box merging)
# Fix 3: min_track_age_for_appear → 2 with area guard (flicker recovery)
# Fix 4: Edge-region area exception (truncated plants at frame edges)
# Fix 5: Elliptical dedup X=60 Y=30 (stop dedup from eating adjacent plants)
#
# Architecture: model.track() with BoT-SORT → per-track crossing / late / appear events

import cv2, csv, json, shutil, subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ═══════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════
MODEL_WEIGHTS       = "plant_finetuned.pt"
TARGET_CLASS_ID     = 0                  # fine-tuned: class 0 = plant
CLASS_CANDIDATES    = [0]

CONF_THRESHOLD      = 0.15
IOU_THRESHOLD       = 0.35              # Fix 2  (was 0.55)
MIN_BOX_AREA        = 450
MIN_BOX_HEIGHT      = 18
TRACKER_CFG         = "botsort.yaml"

SCAN_Y_RATIO        = 0.72              # scan line at 72 % of frame height
CROSSING_BAND_HALF  = 12                # ±12 px around scan line
EVENT_MEMORY_FRAMES = 10                # dedup time window (frames)
DEDUP_RADIUS_X      = 60               # Fix 5  (was circular 45)
DEDUP_RADIUS_Y      = 30               # Fix 5

MIN_TRACK_AGE_LATE    = 4               # frames before late-entry fires
MIN_TRACK_AGE_APPEAR  = 3              # min track age for appear recovery
MIN_AREA_SHORT_TRACK  = 900            # Fix 3  area guard for age ≤ 2
APPEAR_MIN_APPROACH   = 250            # appear only fires if track got within
                                       # this many px of scan_y (prevents
                                       # far-field double-counts with cross)

MIN_TRACK_AGE_FAR     = 8              # Fix A  min frames for far-field appear
FAR_MIN_AVG_CONF      = 0.50           # Fix A  avg confidence for far-field
FAR_LANE_DEDUP_X      = 60             # Fix A  same-lane X proximity guard

EDGE_MARGIN         = 40               # Fix 4  pixels from frame edge
MIN_BOX_AREA_EDGE   = 200              # Fix 4  relaxed threshold for edge boxes

SCAN_CONF           = 0.05              # confidence used in class auto-select scan

OUTPUT_RAW   = "output_raw.mp4"
OUTPUT_FINAL = "output_colab.mp4"
SUMMARY_JSON = "count_summary.json"
EVENT_CSV    = "event_audit.csv"
MISSED_CSV   = "missed_tracks.csv"


# ═══════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════
def pt_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def build_lane_gates(w, h):
    # Fix A: y_top at 2% and wider top boundaries for perspective convergence.
    # At far-field distances the two crop rows converge toward the center of
    # the frame, so the gate trapezoids need to be wider at the top to capture
    # upper-zone (far-field) detections that the original 0.30/0.70 missed.
    # Bottom edges stay close to original values for near-field accuracy.
    yt, yb = int(h * 0.02), int(h * 0.98)
    L = np.array([[0, yt], [int(w * 0.45), yt],
                  [int(w * 0.48), yb], [0, yb]], np.int32)
    R = np.array([[int(w * 0.55), yt], [w - 1, yt],
                  [w - 1, yb], [int(w * 0.52), yb]], np.int32)
    return {"left": L, "right": R}


def get_lane(anchor, gates):
    for name in ("left", "right"):
        if pt_in_poly(anchor, gates[name]):
            return name
    return None


# ═══════════════════════════════════════════════
#  FIX 5 – ELLIPTICAL DEDUP
# ═══════════════════════════════════════════════
def is_dup(evt, history, fidx):
    for p in history:
        if fidx - p["frame"] > EVENT_MEMORY_FRAMES:
            continue
        if p["lane"] != evt["lane"]:
            continue
        dx = abs(evt["ax"] - p["ax"])
        dy = abs(evt["ay"] - p["ay"])
        if (dx / DEDUP_RADIUS_X) ** 2 + (dy / DEDUP_RADIUS_Y) ** 2 < 1.0:
            return True
    return False


def is_lane_dup(lane, ax, events, radius_x=None):
    """Fix A: prevent far-field events from double-counting a plant already
    registered via cross/appear in the same lane at a similar X position.
    Unlike is_dup (which uses frame window + elliptical distance), this
    checks ALL prior events regardless of frame distance, since the same
    plant can appear at very different Y values between far-field and
    near-field."""
    if radius_x is None:
        radius_x = FAR_LANE_DEDUP_X
    for e in events:
        if e["lane"] != lane:
            continue
        if abs(e["anchor_x"] - ax) < radius_x:
            return True
    return False


# ═══════════════════════════════════════════════
#  CLASS AUTO-SELECTION
# ═══════════════════════════════════════════════
def pick_class(model, vpath, gates):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return 58, {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // 20)
    score = defaultdict(float)
    for i in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok:
            continue
        r = model.predict(frm, conf=SCAN_CONF, iou=0.55, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for bx, ci, cf in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.cls.int().cpu().tolist(),
                              r.boxes.conf.cpu().numpy()):
            if int(ci) not in CLASS_CANDIDATES:
                continue
            x1, y1, x2, y2 = map(int, bx)
            anc = (int((x1 + x2) / 2), int(y2))
            if get_lane(anc, gates) is not None:
                score[int(ci)] += float(cf)
    cap.release()
    if not score:
        return 58, {}
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], {k: round(v) for k, v in ranked[:6]}


# ═══════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════
def process_video(input_path):
    if not input_path:
        raise ValueError("INPUT_VIDEO is not set. Run Cell 2 first.")

    print("Loading model …")
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gates  = build_lane_gates(W, H)
    scan_y = int(H * SCAN_Y_RATIO)

    # --- class auto-selection ---
    cls_id = TARGET_CLASS_ID
    cls_scores = {}
    if cls_id is None:
        cls_id, cls_scores = pick_class(model, input_path, gates)
    cls_name = model.names.get(cls_id, str(cls_id))
    print(f"Target class: {cls_id} ({cls_name})")
    if cls_scores:
        print(f"Class scores: {cls_scores}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(OUTPUT_RAW,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # --- state ---
    trk     = {}           # tid → history dict
    counted = set()
    events  = []           # registered events (→ audit CSV)
    recent  = []           # sliding window for dedup
    Lc = Rc = 0            # lane counters

    def fire(reason, fi, lane, tid, ax, ay, bbox, conf):
        nonlocal Lc, Rc
        candidate = {"frame": fi, "lane": lane, "ax": ax, "ay": ay}
        if is_dup(candidate, recent, fi):
            return False
        counted.add(tid)
        if lane == "left":
            Lc += 1
        else:
            Rc += 1
        row = {
            "frame": fi, "lane": lane, "track_id": int(tid),
            "anchor_x": ax, "anchor_y": ay, "event_reason": reason,
            "bbox_x1": bbox[0], "bbox_y1": bbox[1],
            "bbox_x2": bbox[2], "bbox_y2": bbox[3],
            "conf": round(conf, 4), "count_at_event": Lc + Rc,
        }
        events.append(row)
        recent.append({"frame": fi, "lane": lane, "ax": ax, "ay": ay})
        return True

    fidx = 0
    print(f"Processing {N} frames @ {W}x{H} …")

    # --- frame loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        res = model.track(
            frame, persist=True, tracker=TRACKER_CFG,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[cls_id], verbose=False,
        )[0]

        if (res.boxes is not None
                and res.boxes.id is not None
                and len(res.boxes) > 0):

            boxes = res.boxes.xyxy.cpu().numpy()
            tids  = res.boxes.id.int().cpu().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()

            for bx, tid, cf in zip(boxes, tids, confs):
                x1, y1, x2, y2 = map(int, bx)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                h    = max(0, y2 - y1)

                # Fix 4: relaxed area for edge-truncated boxes
                edge = (x1 < EDGE_MARGIN) or (x2 > W - EDGE_MARGIN)
                eff_min_area = MIN_BOX_AREA_EDGE if edge else MIN_BOX_AREA
                if area < eff_min_area or h < MIN_BOX_HEIGHT:
                    continue

                ax = int((x1 + x2) / 2)
                ay = int(y2)
                lane = get_lane((ax, ay), gates)
                if lane is None:
                    continue

                # --- track bookkeeping ---
                if tid not in trk:
                    trk[tid] = dict(frames=[], axs=[], ays=[], lanes=[],
                                    confs=[], bboxes=[], min_y=ay)
                t = trk[tid]
                t["frames"].append(fidx)
                t["axs"].append(ax)
                t["ays"].append(ay)
                t["lanes"].append(lane)
                t["confs"].append(cf)
                t["bboxes"].append((x1, y1, x2, y2))
                t["min_y"] = min(t["min_y"], ay)

                # --- crossing / late event check ---
                if tid not in counted:
                    was_above   = t["min_y"] < (scan_y - CROSSING_BAND_HALF)
                    in_or_below = ay >= (scan_y - CROSSING_BAND_HALF)

                    if was_above and in_or_below and len(t["frames"]) >= 2:
                        fire("cross", fidx, lane, tid,
                             ax, ay, (x1, y1, x2, y2), cf)
                    elif (not was_above
                          and len(t["frames"]) >= MIN_TRACK_AGE_LATE):
                        fire("late", fidx, lane, tid,
                             ax, ay, (x1, y1, x2, y2), cf)

                # --- draw detection box ---
                clr = (0, 220, 0) if lane == "left" else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                cv2.circle(frame, (ax, ay), 4, (0, 255, 255), -1)
                tag = f"T{tid}"
                if tid in counted:
                    tag += " ok"
                cv2.putText(frame, tag, (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1)

        # --- overlays ---
        for ln, poly in gates.items():
            ov = frame.copy()
            cv2.fillPoly(ov, [poly], (40, 40, 180))
            frame = cv2.addWeighted(ov, 0.10, frame, 0.90, 0)
            cv2.polylines(frame, [poly], True, (70, 70, 255), 2)

        cv2.line(frame, (0, scan_y), (W - 1, scan_y), (80, 255, 80), 2)
        cv2.line(frame, (0, scan_y - CROSSING_BAND_HALF),
                 (W - 1, scan_y - CROSSING_BAND_HALF), (80, 180, 80), 1)
        cv2.line(frame, (0, scan_y + CROSSING_BAND_HALF),
                 (W - 1, scan_y + CROSSING_BAND_HALF), (80, 180, 80), 1)

        # event markers (color-coded by reason)
        for e in events:
            if abs(e["frame"] - fidx) < 18:
                ex, ey = e["anchor_x"], e["anchor_y"]
                ec = {"cross": (0, 0, 255), "late": (0, 165, 255),
                      "appear": (255, 0, 255),
                      "far_appear": (255, 255, 0)}.get(e["event_reason"],
                                                        (0, 0, 255))
                cv2.circle(frame, (ex, ey), 10, ec, 2)
                cv2.putText(frame, f"#{e['count_at_event']}",
                            (ex + 12, ey - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, ec, 1)

        # HUD
        tot = Lc + Rc
        cv2.rectangle(frame, (8, 8), (700, 108), (0, 0, 0), -1)
        cv2.putText(frame, f"Plants: {tot}",
                    (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"L:{Lc}  R:{Rc}",
                    (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2)
        cv2.putText(frame,
                    f"cls:{cls_id} iou:{IOU_THRESHOLD} "
                    f"band:{CROSSING_BAND_HALF * 2}px botsort",
                    (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 255, 180), 1)
        out.write(frame)

        if fidx % 30 == 0:
            print(f"  {fidx}/{N}  plants={tot}")

    # ── Fix 3: appear recovery for uncounted tracks ──
    # Proximity guard: only fire if max anchor_y reached within
    # APPEAR_MIN_APPROACH px of scan_y.  Without this, far-field tracks
    # (ay << scan_y) get counted here AND again by cross as they approach.
    for tid, t in trk.items():
        if tid in counted:
            continue
        age = len(t["frames"])
        if age < MIN_TRACK_AGE_APPEAR:
            continue
        last_bb = t["bboxes"][-1]
        last_area = (last_bb[2] - last_bb[0]) * (last_bb[3] - last_bb[1])
        if age <= 2 and last_area < MIN_AREA_SHORT_TRACK:
            continue
        max_ay = max(t["ays"])
        if max_ay < scan_y - APPEAR_MIN_APPROACH:
            continue
        mid = age // 2
        fire("appear", t["frames"][mid], t["lanes"][mid], tid,
             t["axs"][mid], t["ays"][mid], t["bboxes"][mid], t["confs"][mid])

    # ── Fix A (v2): far_appear recovery for stable far-field tracks ──
    # These tracks never descended close to the scan line, so the normal
    # appear path (APPEAR_MIN_APPROACH guard) rejected them.
    #
    # CRITICAL: far_appear MUST be cross-checked against ALL prior events
    # (cross/late/appear), not just other far_appear events.  As the camera
    # walks forward, the same physical plant transitions from far-field
    # (top of frame) to near-field (crosses scan line).  BoT-SORT often
    # assigns different track IDs at these two moments.  Without cross-dedup,
    # the plant gets counted twice: once via far_appear and once via cross.
    #
    # The duplicate analysis confirmed that 15/20 far_appear events were
    # duplicates with dx < 40px vs the corresponding cross/appear event.
    # Therefore we check lane + X proximity against ALL registered events.
    far_events = []
    for tid, t in trk.items():
        if tid in counted:
            continue
        age = len(t["frames"])
        if age < MIN_TRACK_AGE_FAR:
            continue
        avg_conf = sum(t["confs"]) / len(t["confs"])
        if avg_conf < FAR_MIN_AVG_CONF:
            continue
        max_ay = max(t["ays"])
        if max_ay >= scan_y - APPEAR_MIN_APPROACH:
            continue  # handled by appear path above
        mid = age // 2
        lane = t["lanes"][mid]
        ax   = t["axs"][mid]
        ay   = t["ays"][mid]
        f_start, f_end = t["frames"][0], t["frames"][-1]

        # Cross-dedup against ALL prior registered events (any reason).
        # A far-field detection at x=263 that matches a cross event at
        # x=239 in the same lane (dx=24) is almost certainly the same plant.
        if is_lane_dup(lane, ax, events, radius_x=FAR_LANE_DEDUP_X):
            continue

        # Also dedup against other far_appear events (same lane, close X,
        # AND overlapping frame ranges).
        dup = False
        for fe in far_events:
            if fe["lane"] != lane:
                continue
            if abs(fe["anchor_x"] - ax) > FAR_LANE_DEDUP_X:
                continue
            if f_start <= fe["f_end"] and f_end >= fe["f_start"]:
                dup = True
                break
        if dup:
            continue

        if fire("far_appear", t["frames"][mid], lane, tid,
                ax, ay, t["bboxes"][mid], t["confs"][mid]):
            far_events.append({"lane": lane, "anchor_x": ax,
                               "f_start": f_start, "f_end": f_end})


    cap.release()
    out.release()
    tot = Lc + Rc

    # ═══ FIX 1: AUDIT CSVs ═══════════════════════
    evt_hdr = ["frame", "lane", "track_id", "anchor_x", "anchor_y",
               "event_reason", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
               "conf", "count_at_event"]
    with open(EVENT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=evt_hdr)
        w.writeheader()
        w.writerows(events)
    print(f"\nAudit  -> {EVENT_CSV}  ({len(events)} events)")

    missed = []
    for tid, t in trk.items():
        if tid in counted:
            continue
        crossed = any(abs(y - scan_y) <= CROSSING_BAND_HALF for y in t["ays"])
        missed.append({
            "track_id": int(tid),
            "frames_seen": len(t["frames"]),
            "first_frame": t["frames"][0],
            "last_frame": t["frames"][-1],
            "min_y": min(t["ays"]),
            "max_y": max(t["ays"]),
            "crossed_band": crossed,
            "lane": t["lanes"][-1],
            "avg_conf": round(sum(t["confs"]) / len(t["confs"]), 4),
        })
    m_hdr = ["track_id", "frames_seen", "first_frame", "last_frame",
             "min_y", "max_y", "crossed_band", "lane", "avg_conf"]
    with open(MISSED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m_hdr)
        w.writeheader()
        w.writerows(missed)
    print(f"Missed -> {MISSED_CSV}  ({len(missed)} uncounted tracks)")

    # --- re-encode ---
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", OUTPUT_RAW,
             "-vcodec", "libx264", OUTPUT_FINAL],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copyfile(OUTPUT_RAW, OUTPUT_FINAL)

    # --- summary ---
    reason_dist = {r: sum(1 for e in events if e["event_reason"] == r)
                   for r in ("cross", "late", "appear", "far_appear")}
    summary = {
        "input_video": input_path,
        "model_weights": MODEL_WEIGHTS,
        "target_class_id": cls_id,
        "target_class_name": cls_name,
        "class_auto_scores": cls_scores,
        "left_count": Lc,
        "right_count": Rc,
        "registered_count": tot,
        "scan_y": scan_y,
        "crossing_band_half": CROSSING_BAND_HALF,
        "event_memory_frames": EVENT_MEMORY_FRAMES,
        "dedup_radius_x": DEDUP_RADIUS_X,
        "dedup_radius_y": DEDUP_RADIUS_Y,
        "min_track_age_late": MIN_TRACK_AGE_LATE,
        "min_track_age_appear": MIN_TRACK_AGE_APPEAR,
        "appear_min_approach_px": APPEAR_MIN_APPROACH,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "min_box_area": MIN_BOX_AREA,
        "min_box_area_edge": MIN_BOX_AREA_EDGE,
        "tracker": TRACKER_CFG,
        "output_video": OUTPUT_FINAL,
        "event_audit_csv": EVENT_CSV,
        "missed_tracks_csv": MISSED_CSV,
        "event_count_by_reason": reason_dist,
        "missed_track_count": len(missed),
        "missed_crossed_band": sum(1 for m in missed if m["crossed_band"]),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print(json.dumps(summary, indent=2))
    return summary


# Run pipeline
process_video(INPUT_VIDEO)