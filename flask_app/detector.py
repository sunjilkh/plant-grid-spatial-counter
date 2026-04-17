"""
Plant Detector - Inference wrapper for image and video plant detection.

Wraps the best trained YOLO model and provides clean detection APIs
for both single images and video files.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PlantDetector:
    """Unified plant detection for images and videos."""

    def __init__(self, model_path=None, conf=0.25, iou=0.45):
        if model_path is None:
            model_path = self._find_best_model()

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf = conf
        self.iou = iou

        # Get model info
        self.total_params = sum(p.numel() for p in self.model.model.parameters())
        self.class_names = self.model.names

    def _find_best_model(self):
        """Auto-discover best model from research results or fallback."""
        project_root = Path(__file__).parent.parent.resolve()

        # Check research results
        best_json = project_root / "research" / "results" / "best_model.json"
        if best_json.exists():
            import json
            with open(best_json) as f:
                info = json.load(f)
            weights = info.get("best_weights", "")
            if os.path.exists(weights):
                return weights

        # Fallback: check for ablation models
        ablation_dir = project_root / "runs" / "ablation"
        if ablation_dir.exists():
            for variant in ["yolov8l", "yolov8m", "yolov8s", "yolov8n"]:
                p = ablation_dir / variant / "weights" / "best.pt"
                if p.exists():
                    return str(p)

        # Fallback: plant_finetuned.pt
        finetuned = project_root / "plant_finetuned.pt"
        if finetuned.exists():
            return str(finetuned)

        # Last resort: pretrained
        return "yolov8m.pt"

    def get_model_info(self):
        """Return model metadata."""
        file_size = os.path.getsize(self.model_path) / (1024 * 1024)
        return {
            "model_path": str(self.model_path),
            "model_name": Path(self.model_path).stem,
            "total_params": self.total_params,
            "params_millions": round(self.total_params / 1e6, 2),
            "file_size_mb": round(file_size, 2),
            "class_names": self.class_names,
            "conf_threshold": self.conf,
            "iou_threshold": self.iou,
        }

    def process_image(self, image_path, output_dir, job_id):
        """Detect plants in a single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        H, W = img.shape[:2]

        t_start = time.time()
        results = self.model.predict(
            img, conf=self.conf, iou=self.iou, verbose=False
        )[0]
        inference_ms = (time.time() - t_start) * 1000

        detections = []
        annotated = img.copy()
        confs = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.int().cpu().tolist(),
            ):
                x1, y1, x2, y2 = map(int, box)
                c = float(conf)
                confs.append(c)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(c, 4),
                    "class_id": int(cls),
                    "class_name": self.class_names.get(int(cls), str(cls)),
                })

                # Draw on image
                color = (0, 220, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"Plant {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add count HUD
        count = len(detections)
        cv2.rectangle(annotated, (5, 5), (300, 55), (0, 0, 0), -1)
        cv2.putText(annotated, f"Plants detected: {count}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)

        # Save annotated image
        output_name = f"{job_id}_result.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, annotated)

        # Confidence stats
        conf_stats = {}
        if confs:
            confs.sort()
            conf_stats = {
                "mean": round(sum(confs) / len(confs), 4),
                "min": round(min(confs), 4),
                "max": round(max(confs), 4),
                "median": round(confs[len(confs) // 2], 4),
            }

        return {
            "plant_count": count,
            "detections": detections,
            "inference_time_ms": round(inference_ms, 2),
            "output_path": output_path,
            "confidence_stats": conf_stats,
            "model_info": self.get_model_info(),
        }

    def process_video(self, video_path, output_dir, job_id):
        """Detect and count unique plants in a video using tracking."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_name = f"{job_id}_result.mp4"
        output_path = os.path.join(output_dir, output_name)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        unique_tracks = set()
        all_confs = []
        total_inference_ms = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            t0 = time.time()
            results = self.model.track(
                frame, persist=True, tracker="botsort.yaml",
                conf=self.conf, iou=self.iou, verbose=False
            )[0]
            total_inference_ms += (time.time() - t0) * 1000

            if (results.boxes is not None
                    and results.boxes.id is not None
                    and len(results.boxes) > 0):

                for box, tid, conf, cls in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.id.int().cpu().tolist(),
                    results.boxes.conf.cpu().numpy(),
                    results.boxes.cls.int().cpu().tolist(),
                ):
                    x1, y1, x2, y2 = map(int, box)
                    c = float(conf)
                    unique_tracks.add(int(tid))
                    all_confs.append(c)

                    # Color by track
                    np.random.seed(int(tid) * 7)
                    color = tuple(int(x) for x in np.random.randint(80, 255, 3))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"P{tid} {c:.2f}"
                    cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # HUD
            count = len(unique_tracks)
            cv2.rectangle(frame, (5, 5), (350, 55), (0, 0, 0), -1)
            cv2.putText(frame, f"Unique plants: {count}  Frame: {frame_count}/{total_frames}",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            out.write(frame)

        cap.release()
        out.release()

        # Re-encode with ffmpeg if available
        if shutil.which("ffmpeg"):
            import subprocess
            final = output_path.replace(".mp4", "_h264.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", output_path,
                            "-vcodec", "libx264", final],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(output_path)
            os.rename(final, output_path)

        # Confidence stats
        conf_stats = {}
        if all_confs:
            all_confs.sort()
            conf_stats = {
                "mean": round(sum(all_confs) / len(all_confs), 4),
                "min": round(min(all_confs), 4),
                "max": round(max(all_confs), 4),
                "median": round(all_confs[len(all_confs) // 2], 4),
            }

        return {
            "plant_count": len(unique_tracks),
            "inference_time_ms": round(total_inference_ms, 2),
            "output_path": output_path,
            "confidence_stats": conf_stats,
            "total_frames": frame_count,
            "avg_ms_per_frame": round(total_inference_ms / max(1, frame_count), 2),
            "model_info": self.get_model_info(),
        }


import shutil
