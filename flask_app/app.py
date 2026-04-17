"""
Plant Detection Flask App - Production Deployment Server.

Provides a web interface for uploading images/videos and detecting
unique plants using the best trained YOLOv8 model.

Usage:
    .venv\Scripts\python.exe flask_app\app.py
"""

import os
import sys
import json
import time
import uuid
import shutil
from pathlib import Path

from flask import Flask, request, render_template, jsonify, send_from_directory, url_for

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from detector import PlantDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max upload
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "static" / "uploads"
app.config["RESULTS_FOLDER"] = Path(__file__).parent / "static" / "results"

# Ensure directories exist
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["RESULTS_FOLDER"].mkdir(parents=True, exist_ok=True)

# Initialize detector
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = PlantDetector()
    return detector


ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE | ALLOWED_VIDEO


def is_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_VIDEO


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE | ALLOWED_VIDEO)}"}), 400

    # Save upload with unique name
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower()
    input_filename = f"{job_id}_input{ext}"
    input_path = app.config["UPLOAD_FOLDER"] / input_filename
    file.save(str(input_path))

    try:
        det = get_detector()

        if is_video(file.filename):
            result = det.process_video(str(input_path), str(app.config["RESULTS_FOLDER"]), job_id)
        else:
            result = det.process_image(str(input_path), str(app.config["RESULTS_FOLDER"]), job_id)

        # Build response
        response = {
            "success": True,
            "job_id": job_id,
            "input_type": "video" if is_video(file.filename) else "image",
            "plant_count": result["plant_count"],
            "detections": result.get("detections", []),
            "inference_time_ms": result["inference_time_ms"],
            "confidence_stats": result.get("confidence_stats", {}),
            "model_info": result.get("model_info", {}),
        }

        if result.get("output_path"):
            output_name = Path(result["output_path"]).name
            response["output_url"] = url_for("static",
                                            filename=f"results/{output_name}")

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    det = get_detector()
    return jsonify(det.get_model_info())


if __name__ == "__main__":
    print("=" * 50)
    print("  Plant Detection Server")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
