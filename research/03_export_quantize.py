"""
03_export_quantize.py - Export best model to ONNX and TFLite formats.

Exports the best model from ablation study to:
- ONNX FP32
- ONNX FP16
- TFLite FP32
- TFLite FP16
- TFLite INT8 (with calibration on training data)

Records model size for each format.

Usage: .venv\Scripts\python.exe research\03_export_quantize.py
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

DATA_YAML   = str(PROJECT_ROOT / "finetune_data" / "data.yaml")
RESULTS_DIR = Path(__file__).parent / "results"
EXPORT_DIR  = PROJECT_ROOT / "exported_models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_best_model():
    """Find the best model from ablation results."""
    best_json = RESULTS_DIR / "best_model.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        weights = info["best_weights"]
        if os.path.exists(weights):
            return info["variant"], weights

    # Fallback: check evaluation results
    eval_json = RESULTS_DIR / "evaluation_results.json"
    if eval_json.exists():
        with open(eval_json) as f:
            results = json.load(f)
        valid = [r for r in results if "error" not in r and r.get("mAP50")]
        if valid:
            best = max(valid, key=lambda x: x["mAP50"])
            return best["variant"], best["weights_path"]

    # Fallback: use existing finetuned model
    finetuned = PROJECT_ROOT / "plant_finetuned.pt"
    if finetuned.exists():
        return "yolov8m_finetuned", str(finetuned)

    raise FileNotFoundError("No trained model found. Run 01_train_ablation.py first.")


def export_format(model_path, fmt, half=False, int8=False, name_suffix=""):
    """Export model to a specific format and return info."""
    print(f"\n  Exporting to {fmt} (half={half}, int8={int8})...")

    model = YOLO(model_path)

    export_args = dict(format=fmt, imgsz=640)
    if half:
        export_args["half"] = True
    if int8:
        export_args["int8"] = True
        export_args["data"] = DATA_YAML

    t_start = time.time()
    exported_path = model.export(**export_args)
    export_time = time.time() - t_start

    # Get file size
    if exported_path and os.path.exists(exported_path):
        size_mb = os.path.getsize(exported_path) / (1024 * 1024)
    elif isinstance(exported_path, str) and os.path.isdir(exported_path):
        # TFLite models may be in a directory
        total = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, dn, fnames in os.walk(exported_path)
                    for f in fnames)
        size_mb = total / (1024 * 1024)
    else:
        size_mb = None

    result = {
        "format": f"{fmt}{'_fp16' if half else ''}{'_int8' if int8 else ''}",
        "exported_path": str(exported_path) if exported_path else None,
        "size_mb": round(size_mb, 2) if size_mb else None,
        "export_time_sec": round(export_time, 1),
    }

    print(f"    Path: {exported_path}")
    print(f"    Size: {result['size_mb']} MB")
    print(f"    Time: {result['export_time_sec']}s")

    return result


def main():
    print("=" * 70)
    print("  MODEL EXPORT & QUANTIZATION")
    print("=" * 70)

    variant, weights_path = find_best_model()
    print(f"  Best model: {variant}")
    print(f"  Weights: {weights_path}")

    # Original model size
    orig_size = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"  Original size: {orig_size:.2f} MB")

    all_exports = []

    # Record original PyTorch model
    all_exports.append({
        "format": "pytorch_fp32",
        "exported_path": weights_path,
        "size_mb": round(orig_size, 2),
        "export_time_sec": 0,
    })

    # Export formats
    exports_to_run = [
        ("onnx",   False, False),   # ONNX FP32
        ("onnx",   True,  False),   # ONNX FP16
    ]

    # TFLite exports need tensorflow - check first
    try:
        import tensorflow
        exports_to_run.extend([
            ("tflite", False, False),  # TFLite FP32
            ("tflite", True,  False),  # TFLite FP16
            ("tflite", False, True),   # TFLite INT8
        ])
        print("  TensorFlow found - will export TFLite variants")
    except ImportError:
        print("  TensorFlow not installed - skipping TFLite exports")
        print("  Install with: pip install tensorflow")

    for fmt, half, int8 in exports_to_run:
        try:
            result = export_format(weights_path, fmt, half=half, int8=int8)
            result["source_variant"] = variant
            all_exports.append(result)
        except Exception as e:
            print(f"  ERROR exporting {fmt} (half={half}, int8={int8}): {e}")
            all_exports.append({
                "format": f"{fmt}{'_fp16' if half else ''}{'_int8' if int8 else ''}",
                "error": str(e),
                "source_variant": variant,
            })

    # ─── Save results ────────────────────────────────────
    json_path = RESULTS_DIR / "export_results.json"
    with open(json_path, "w") as f:
        json.dump(all_exports, f, indent=2, default=str)

    csv_path = RESULTS_DIR / "export_results.csv"
    fieldnames = sorted(set(k for r in all_exports for k in r.keys()))
    priority = ["format", "size_mb", "export_time_sec", "exported_path"]
    ordered = [c for c in priority if c in fieldnames]
    ordered += [c for c in fieldnames if c not in ordered]

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_exports)

    print(f"\nExport results saved to: {json_path}")
    print(f"Export results saved to: {csv_path}")

    # ─── Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Format':<18} {'Size(MB)':>10} {'Reduction':>10}")
    print("-" * 40)
    for r in all_exports:
        if "error" in r:
            print(f"{r['format']:<18} ERROR")
            continue
        reduction = f"{(1 - r['size_mb']/orig_size)*100:.1f}%" if r.get("size_mb") else "-"
        print(f"{r['format']:<18} {r.get('size_mb', '-'):>10} {reduction:>10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
