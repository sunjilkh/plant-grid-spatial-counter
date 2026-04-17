"""
01_train_ablation.py - Ablation Study: Train YOLOv8 variants on plant data.

Trains YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l with identical hyperparameters.
Records per-epoch metrics and saves results to research/results/.

Usage: .venv\Scripts\python.exe research\01_train_ablation.py
"""

import json
import time
import csv
import os
import sys
from pathlib import Path

# Ensure we can import from project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────
DATA_YAML    = str(PROJECT_ROOT / "finetune_data" / "data.yaml")
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ablation variants
VARIANTS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

# Shared hyperparameters (held constant for fair comparison)
TRAIN_ARGS = dict(
    data       = DATA_YAML,
    epochs     = 25,
    imgsz      = 640,
    batch      = 4,
    patience   = 10,
    device     = "cpu",
    workers    = 0,
    verbose    = True,
    save       = True,
    plots      = True,
    val        = True,
)


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return total, trainable


def train_variant(variant_name):
    """Train a single YOLO variant and return metrics."""
    print("=" * 70)
    print(f"  TRAINING: {variant_name}")
    print("=" * 70)

    pretrained = f"{variant_name}.pt"
    project_dir = str(PROJECT_ROOT / "runs" / "ablation")
    run_name = variant_name

    model = YOLO(pretrained)
    total_params, train_params = count_params(model)
    print(f"  Parameters: {total_params:,} total, {train_params:,} trainable")

    t_start = time.time()
    results = model.train(
        **TRAIN_ARGS,
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )
    t_elapsed = time.time() - t_start

    # Collect final metrics from results
    metrics = {
        "variant": variant_name,
        "total_params": total_params,
        "trainable_params": train_params,
        "train_time_sec": round(t_elapsed, 1),
        "epochs_completed": results.epoch + 1 if hasattr(results, 'epoch') else TRAIN_ARGS["epochs"],
        "final_mAP50": None,
        "final_mAP50_95": None,
        "final_precision": None,
        "final_recall": None,
        "final_box_loss": None,
        "best_weights": str(Path(project_dir) / run_name / "weights" / "best.pt"),
        "last_weights": str(Path(project_dir) / run_name / "weights" / "last.pt"),
    }

    # Try to extract metrics from results object
    try:
        if hasattr(results, 'results_dict'):
            rd = results.results_dict
            metrics["final_mAP50"] = round(rd.get("metrics/mAP50(B)", 0), 4)
            metrics["final_mAP50_95"] = round(rd.get("metrics/mAP50-95(B)", 0), 4)
            metrics["final_precision"] = round(rd.get("metrics/precision(B)", 0), 4)
            metrics["final_recall"] = round(rd.get("metrics/recall(B)", 0), 4)
            metrics["final_box_loss"] = round(rd.get("val/box_loss", 0), 4)
    except Exception as e:
        print(f"  Warning: Could not extract metrics: {e}")

    # Try to get F1
    if metrics["final_precision"] and metrics["final_recall"]:
        p, r = metrics["final_precision"], metrics["final_recall"]
        metrics["final_f1"] = round(2 * p * r / (p + r + 1e-8), 4)
    else:
        metrics["final_f1"] = None

    # FLOPs
    try:
        model_info = model.info()
        if isinstance(model_info, tuple) and len(model_info) >= 2:
            metrics["flops_G"] = model_info[1]
    except:
        metrics["flops_G"] = None

    return metrics


def main():
    print("=" * 70)
    print("  PLANT DETECTION ABLATION STUDY")
    print("  Variants:", ", ".join(VARIANTS))
    print("  Epochs:", TRAIN_ARGS["epochs"])
    print("  Image size:", TRAIN_ARGS["imgsz"])
    print("  Dataset:", DATA_YAML)
    print("=" * 70)

    all_results = []

    for variant in VARIANTS:
        try:
            metrics = train_variant(variant)
            all_results.append(metrics)
            print(f"\n  {variant} COMPLETE:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
            print()
        except Exception as e:
            print(f"\n  ERROR training {variant}: {e}")
            all_results.append({"variant": variant, "error": str(e)})

    # ─── Save results to CSV ─────────────────────────────
    csv_path = RESULTS_DIR / "ablation_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        # Merge all keys
        for r in all_results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to: {csv_path}")

    # ─── Save results to JSON ────────────────────────────
    json_path = RESULTS_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")

    # ─── Print summary table ─────────────────────────────
    print("\n" + "=" * 90)
    print("ABLATION STUDY SUMMARY")
    print("=" * 90)
    header = f"{'Variant':<12} {'Params':>10} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'F1':>8} {'Time(s)':>9}"
    print(header)
    print("-" * 90)
    for r in all_results:
        if "error" in r:
            print(f"{r['variant']:<12} ERROR: {r['error']}")
            continue
        print(f"{r['variant']:<12} {r['total_params']:>10,} "
              f"{r.get('final_mAP50', 'N/A'):>8} "
              f"{r.get('final_mAP50_95', 'N/A'):>10} "
              f"{r.get('final_precision', 'N/A'):>8} "
              f"{r.get('final_recall', 'N/A'):>8} "
              f"{r.get('final_f1', 'N/A'):>8} "
              f"{r.get('train_time_sec', 'N/A'):>9}")
    print("=" * 90)

    # ─── Determine best model ────────────────────────────
    valid = [r for r in all_results if "error" not in r and r.get("final_mAP50")]
    if valid:
        best = max(valid, key=lambda x: x["final_mAP50"])
        print(f"\nBEST MODEL: {best['variant']} (mAP50={best['final_mAP50']})")
        print(f"  Weights: {best['best_weights']}")

        # Save best model path for next scripts
        with open(RESULTS_DIR / "best_model.json", "w") as f:
            json.dump({"variant": best["variant"],
                        "best_weights": best["best_weights"],
                        "mAP50": best["final_mAP50"]}, f, indent=2)


if __name__ == "__main__":
    main()
