"""
02_evaluate_models.py - Detailed evaluation of all trained ablation models.

Runs validation on each model checkpoint and records:
- Precision, Recall, F1, mAP50, mAP50-95
- Inference time (preprocess + inference + postprocess)
- Model parameters, FLOPs
- Per-class metrics

Usage: .venv\Scripts\python.exe research\02_evaluate_models.py
"""

import json
import time
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

DATA_YAML   = str(PROJECT_ROOT / "finetune_data" / "data.yaml")
RESULTS_DIR = Path(__file__).parent / "results"
ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"

VARIANTS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]


def evaluate_model(variant, weights_path):
    """Run full validation on a model and collect all metrics."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {variant}")
    print(f"  Weights: {weights_path}")
    print(f"{'='*60}")

    model = YOLO(weights_path)

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    # Model file size
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    # Run validation
    t_start = time.time()
    val_results = model.val(data=DATA_YAML, imgsz=640, batch=4, verbose=True,
                            device="cpu", workers=0)
    val_time = time.time() - t_start

    # Extract metrics
    metrics = {
        "variant": variant,
        "weights_path": str(weights_path),
        "file_size_mb": round(file_size_mb, 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_millions": round(total_params / 1e6, 2),
    }

    # Box metrics
    try:
        box = val_results.box
        metrics["precision"] = round(float(box.mp), 4)
        metrics["recall"] = round(float(box.mr), 4)
        metrics["mAP50"] = round(float(box.map50), 4)
        metrics["mAP50_95"] = round(float(box.map), 4)

        p, r = metrics["precision"], metrics["recall"]
        metrics["f1"] = round(2 * p * r / (p + r + 1e-8), 4)
    except Exception as e:
        print(f"  Warning extracting box metrics: {e}")

    # Speed metrics
    try:
        speed = val_results.speed
        metrics["preprocess_ms"] = round(speed.get("preprocess", 0), 2)
        metrics["inference_ms"] = round(speed.get("inference", 0), 2)
        metrics["postprocess_ms"] = round(speed.get("postprocess", 0), 2)
        metrics["total_per_image_ms"] = round(
            metrics["preprocess_ms"] + metrics["inference_ms"] + metrics["postprocess_ms"], 2)
    except Exception as e:
        print(f"  Warning extracting speed: {e}")

    metrics["val_time_sec"] = round(val_time, 1)

    # FLOPs
    try:
        info = model.info()
        if isinstance(info, tuple) and len(info) >= 2:
            metrics["flops_G"] = round(info[1], 2)
    except:
        metrics["flops_G"] = None

    # Confidence analysis - run predict on val images and collect conf stats
    try:
        val_img_dir = PROJECT_ROOT / "finetune_data" / "images" / "val"
        all_confs = []
        for img_path in val_img_dir.glob("*.jpg"):
            res = model.predict(str(img_path), conf=0.15, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                confs = res.boxes.conf.cpu().numpy().tolist()
                all_confs.extend(confs)

        if all_confs:
            all_confs.sort()
            metrics["conf_mean"] = round(sum(all_confs) / len(all_confs), 4)
            metrics["conf_median"] = round(all_confs[len(all_confs) // 2], 4)
            metrics["conf_min"] = round(min(all_confs), 4)
            metrics["conf_max"] = round(max(all_confs), 4)
            metrics["conf_p25"] = round(all_confs[len(all_confs) // 4], 4)
            metrics["conf_p75"] = round(all_confs[3 * len(all_confs) // 4], 4)
            metrics["total_detections_val"] = len(all_confs)
    except Exception as e:
        print(f"  Warning in confidence analysis: {e}")

    return metrics


def main():
    print("=" * 70)
    print("  MODEL EVALUATION - ALL ABLATION VARIANTS")
    print("=" * 70)

    all_results = []

    for variant in VARIANTS:
        weights = ABLATION_DIR / variant / "weights" / "best.pt"
        if not weights.exists():
            print(f"  SKIPPING {variant}: weights not found at {weights}")
            continue
        try:
            metrics = evaluate_model(variant, str(weights))
            all_results.append(metrics)
        except Exception as e:
            print(f"  ERROR evaluating {variant}: {e}")
            all_results.append({"variant": variant, "error": str(e)})

    # ─── Save results ─────────────────────────────────────
    if not all_results:
        print("No models to evaluate. Run 01_train_ablation.py first.")
        return

    csv_path = RESULTS_DIR / "evaluation_results.csv"
    fieldnames = sorted(set(k for r in all_results for k in r.keys()))
    # Put key columns first
    priority = ["variant", "precision", "recall", "f1", "mAP50", "mAP50_95",
                "inference_ms", "total_params", "file_size_mb"]
    ordered = [c for c in priority if c in fieldnames]
    ordered += [c for c in fieldnames if c not in ordered]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    json_path = RESULTS_DIR / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {csv_path}")
    print(f"Results saved to: {json_path}")

    # ─── Summary table ────────────────────────────────────
    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"{'Variant':<10} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7} {'mAP50-95':>9} "
          f"{'Inf(ms)':>8} {'Size(MB)':>9} {'Params(M)':>10} {'ConfMean':>9}")
    print("-" * 100)
    for r in all_results:
        if "error" in r:
            print(f"{r['variant']:<10} ERROR")
            continue
        print(f"{r['variant']:<10} "
              f"{r.get('precision', '-'):>7} "
              f"{r.get('recall', '-'):>7} "
              f"{r.get('f1', '-'):>7} "
              f"{r.get('mAP50', '-'):>7} "
              f"{r.get('mAP50_95', '-'):>9} "
              f"{r.get('inference_ms', '-'):>8} "
              f"{r.get('file_size_mb', '-'):>9} "
              f"{r.get('params_millions', '-'):>10} "
              f"{r.get('conf_mean', '-'):>9}")
    print(f"{'='*100}")

    # Best model
    valid = [r for r in all_results if "error" not in r and r.get("mAP50")]
    if valid:
        best = max(valid, key=lambda x: x["mAP50"])
        print(f"\nBEST MODEL: {best['variant']}")
        print(f"  mAP50={best['mAP50']}  P={best.get('precision')}  "
              f"R={best.get('recall')}  F1={best.get('f1')}")
        print(f"  Inference: {best.get('inference_ms')}ms  Size: {best.get('file_size_mb')}MB")


if __name__ == "__main__":
    main()
