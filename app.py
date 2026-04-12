"""
Plant Counter – Gradio Web App
Upload a garden video → get plant count + annotated video + audit files.

Run locally:
    .venv/Scripts/python.exe app.py

Public internet URL (valid 72 hours):
    Change  demo.launch(share=False)
    to      demo.launch(share=True)
    at the bottom of this file.

Permanent hosting: push this file to a Hugging Face Space (free tier).
"""

import os
import csv
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Resolve model path relative to this file so the app works from any CWD
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
MODEL_WEIGHTS = str(APP_DIR / "plant_finetuned.pt")

# ---------------------------------------------------------------------------
# CONFIGURATION  (identical to notebook Cell 2)
# ---------------------------------------------------------------------------
TARGET_CLASS_ID     = 0
CLASS_CANDIDATES    = [58, 10, 50, 2, 6, 8]

CONF_THRESHOLD      = 0.15
IOU_THRESHOLD       = 0.35
MIN_BOX_AREA        = 450
MIN_BOX_HEIGHT      = 18
TRACKER_CFG         = "botsort.yaml"

SCAN_Y_RATIO        = 0.72
CROSSING_BAND_HALF  = 12
EVENT_MEMORY_FRAMES = 10
DEDUP_RADIUS_X      = 60
DEDUP_RADIUS_Y      = 30

MIN_TRACK_AGE_LATE    = 4
MIN_TRACK_AGE_APPEAR  = 3
MIN_AREA_SHORT_TRACK  = 900
APPEAR_MIN_APPROACH   = 300

FAR_FIELD_MIN_AGE   = 25
FAR_FIELD_MIN_CONF  = 0.60
FAR_NEAR_DEDUP_X    = 50

EDGE_MARGIN         = 40
MIN_BOX_AREA_EDGE   = 200
SCAN_CONF           = 0.05

# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def pt_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def build_lane_gates(w, h):
    yt, yb = int(h * 0.05), int(h * 0.98)
    L = np.array([[0, yt], [int(w * 0.30), yt],
                  [int(w * 0.48), yb], [0, yb]], np.int32)
    R = np.array([[int(w * 0.70), yt], [w - 1, yt],
                  [w - 1, yb], [int(w * 0.52), yb]], np.int32)
    return {"left": L, "right": R}


def get_lane(anchor, gates):
    for name in ("left", "right"):
        if pt_in_poly(anchor, gates[name]):
            return name
    return None


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


# ---------------------------------------------------------------------------
# MAIN PIPELINE  (verbatim from notebook Cell 2, paths replaced with tmpdir)
# ---------------------------------------------------------------------------
def process_video(input_path, tmpdir):
    output_raw   = os.path.join(tmpdir, "output_raw.mp4")
    output_final = os.path.join(tmpdir, "output_colab.mp4")
    summary_json = os.path.join(tmpdir, "count_summary.json")
    event_csv    = os.path.join(tmpdir, "event_audit.csv")
    missed_csv   = os.path.join(tmpdir, "missed_tracks.csv")

    print("Loading model ...")
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

    cls_id = TARGET_CLASS_ID
    cls_scores = {}
    if cls_id is None:
        cls_id, cls_scores = pick_class(model, input_path, gates)
    cls_name = model.names.get(cls_id, str(cls_id))
    print(f"Target class: {cls_id} ({cls_name})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(output_raw,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trk     = {}
    counted = set()
    events  = []
    recent  = []
    Lc = Rc = 0

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
    print(f"Processing {N} frames @ {W}x{H} ...")

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
                edge = (x1 < EDGE_MARGIN) or (x2 > W - EDGE_MARGIN)
                eff_min_area = MIN_BOX_AREA_EDGE if edge else MIN_BOX_AREA
                if area < eff_min_area or h < MIN_BOX_HEIGHT:
                    continue

                ax = int((x1 + x2) / 2)
                ay = int(y2)
                lane = get_lane((ax, ay), gates)
                if lane is None:
                    continue

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

                if (tid not in counted
                        and abs(ay - scan_y) <= CROSSING_BAND_HALF
                        and len(t["frames"]) == 1
                        and cf >= 0.40):
                    fire("snap_cross", fidx, lane, tid,
                         ax, ay, (x1, y1, x2, y2), cf)

                clr = (0, 220, 0) if lane == "left" else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                cv2.circle(frame, (ax, ay), 4, (0, 255, 255), -1)
                tag = f"T{tid}" + (" ok" if tid in counted else "")
                cv2.putText(frame, tag, (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1)

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

        for e in events:
            if abs(e["frame"] - fidx) < 18:
                ex, ey = e["anchor_x"], e["anchor_y"]
                ec = {"cross": (0, 0, 255), "late": (0, 165, 255),
                      "appear": (255, 0, 255)}.get(e["event_reason"],
                                                    (0, 0, 255))
                cv2.circle(frame, (ex, ey), 10, ec, 2)
                cv2.putText(frame, f"#{e['count_at_event']}",
                            (ex + 12, ey - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, ec, 1)

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

    # --- appear recovery ---
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

    # --- far-field recovery ---
    for tid, t in trk.items():
        if tid in counted:
            continue
        age = len(t["frames"])
        if age < FAR_FIELD_MIN_AGE:
            continue
        avg_c = sum(t["confs"]) / age
        if avg_c < FAR_FIELD_MIN_CONF:
            continue
        max_ay = max(t["ays"])
        if max_ay >= scan_y - APPEAR_MIN_APPROACH:
            continue
        lane_counts = {}
        for ln in t["lanes"]:
            lane_counts[ln] = lane_counts.get(ln, 0) + 1
        dom_lane = max(lane_counts, key=lane_counts.get)
        if lane_counts[dom_lane] / age < 0.80:
            continue
        mid = age // 2
        fire("far_appear", t["frames"][mid], dom_lane, tid,
             t["axs"][mid], t["ays"][mid], t["bboxes"][mid], t["confs"][mid])

    # --- cross-tier dedup ---
    cross_late = [e for e in events
                  if e["event_reason"] in ("cross", "late", "snap_cross")]
    to_remove = []
    for i, e in enumerate(events):
        if e["event_reason"] not in ("far_appear", "appear"):
            continue
        for cl in cross_late:
            if cl["lane"] != e["lane"]:
                continue
            if abs(cl["anchor_x"] - e["anchor_x"]) < FAR_NEAR_DEDUP_X:
                to_remove.append(i)
                break
    for idx in sorted(to_remove, reverse=True):
        removed = events.pop(idx)
        if removed["lane"] == "left":
            Lc -= 1
        else:
            Rc -= 1
    tot = Lc + Rc

    cap.release()
    out.release()

    # --- write audit CSVs ---
    evt_hdr = ["frame", "lane", "track_id", "anchor_x", "anchor_y",
               "event_reason", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
               "conf", "count_at_event"]
    with open(event_csv, "w", newline="") as f:
        ew = csv.DictWriter(f, fieldnames=evt_hdr)
        ew.writeheader()
        ew.writerows(events)

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
    with open(missed_csv, "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=m_hdr)
        w2.writeheader()
        w2.writerows(missed)

    # --- re-encode for browser playback ---
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_raw,
             "-vcodec", "libx264", output_final],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copyfile(output_raw, output_final)

    # --- summary JSON ---
    reason_dist = {r: sum(1 for e in events if e["event_reason"] == r)
                   for r in ("cross", "late", "appear", "far_appear",
                             "snap_cross")}
    high_q_missed = sum(1 for m in missed
                        if m["frames_seen"] >= 5 and m["avg_conf"] >= 0.5)
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
        "output_video": output_final,
        "event_audit_csv": event_csv,
        "missed_tracks_csv": missed_csv,
        "far_field_min_age": FAR_FIELD_MIN_AGE,
        "far_field_min_conf": FAR_FIELD_MIN_CONF,
        "far_near_dedup_x": FAR_NEAR_DEDUP_X,
        "event_count_by_reason": reason_dist,
        "missed_track_count": len(missed),
        "missed_crossed_band": sum(1 for m in missed if m["crossed_band"]),
        "high_quality_missed": high_q_missed,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, output_final, summary_json, event_csv, missed_csv


# ---------------------------------------------------------------------------
# GRADIO WRAPPER
# ---------------------------------------------------------------------------
def run_counter(video_path):
    """Gradio entrypoint: video_path is a temp file Gradio wrote for us."""
    if video_path is None:
        return "No video uploaded.", None, None, None, None

    tmpdir = tempfile.mkdtemp(prefix="plant_counter_")
    try:
        summary, vid_out, json_out, evt_csv, mis_csv = process_video(
            video_path, tmpdir
        )
    except Exception as e:
        return f"Error during processing: {e}", None, None, None, None

    s = summary
    count_text = (
        f"TOTAL PLANTS COUNTED: {s['registered_count']}\n"
        f"\n"
        f"Left lane:   {s['left_count']}\n"
        f"Right lane:  {s['right_count']}\n"
        f"\n"
        f"Event breakdown:\n"
        f"  Crossing events : {s['event_count_by_reason'].get('cross', 0)}\n"
        f"  Late-entry      : {s['event_count_by_reason'].get('late', 0)}\n"
        f"  Appear recovery : {s['event_count_by_reason'].get('appear', 0)}\n"
        f"  Far-field       : {s['event_count_by_reason'].get('far_appear', 0)}\n"
        f"  Snap-cross      : {s['event_count_by_reason'].get('snap_cross', 0)}\n"
        f"\n"
        f"Missed tracks (uncounted) : {s['missed_track_count']}\n"
        f"High-quality missed       : {s['high_quality_missed']}\n"
        f"Crossed-band but uncounted: {s['missed_crossed_band']}\n"
        f"\n"
        f"Model: {Path(s['model_weights']).name}  |  "
        f"conf={s['conf_threshold']}  iou={s['iou_threshold']}"
    )

    return count_text, vid_out, json_out, evt_csv, mis_csv


# ---------------------------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------------------------
with gr.Blocks(title="Plant Counter") as demo:
    gr.Markdown(
        """
        # Plant Counter
        Upload a garden walkthrough video and get a precise plant count
        using spatial-position tracking (one position = one plant).

        **Note:** The model is calibrated for two-row nursery setups
        with a forward-walking camera. Processing takes ~60 seconds
        per 200 frames on CPU.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload garden video")
            run_btn = gr.Button("Count Plants", variant="primary")
        with gr.Column(scale=1):
            count_output = gr.Textbox(
                label="Count Result",
                lines=18,
                interactive=False,
            )

    with gr.Row():
        video_output = gr.Video(label="Annotated Output Video")

    with gr.Row():
        json_output  = gr.File(label="Download Summary JSON")
        evt_output   = gr.File(label="Download Event Audit CSV")
        miss_output  = gr.File(label="Download Missed Tracks CSV")

    run_btn.click(
        fn=run_counter,
        inputs=[video_input],
        outputs=[count_output, video_output, json_output,
                 evt_output, miss_output],
    )

# ---------------------------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # share=False  → local only  (http://127.0.0.1:7860)
    # share=True   → public gradio.live URL (valid 72 hours)
    demo.launch(share=False)
