"""
04_benchmark_quant.py - Benchmark all exported model formats.

Runs inference on validation images with each exported format and records:
- Precision, Recall, F1 (via val())
- Inference time per image
- Model file size
- Comparison table

Usage: .venv\Scripts\python.exe research\04_benchmark_quant.py
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


def benchmark_model(model_path, format_name):
    """Run validation using a specific model format and measure performance."""
    print(f"\n  Benchmarking: {format_name}")
    print(f"  Path: {model_path}")

    if not os.path.exists(model_path):
        return {"format": format_name, "error": f"File not found: {model_path}"}

    # File size
    if os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
    else:
        size_mb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, dn, fnames in os.walk(model_path)
                      for f in fnames) / (1024 * 1024)

    model = YOLO(model_path)

    # Run validation
    t_start = time.time()
    val_results = model.val(data=DATA_YAML, imgsz=640, batch=1, verbose=False,
                            device="cpu", workers=0)
    val_time = time.time() - t_start

    result = {
        "format": format_name,
        "model_path": str(model_path),
        "size_mb": round(size_mb, 2),
    }

    # Metrics
    try:
        box = val_results.box
        result["precision"] = round(float(box.mp), 4)
        result["recall"] = round(float(box.mr), 4)
        result["mAP50"] = round(float(box.map50), 4)
        result["mAP50_95"] = round(float(box.map), 4)
        p, r = result["precision"], result["recall"]
        result["f1"] = round(2 * p * r / (p + r + 1e-8), 4)
    except Exception as e:
        print(f"    Warning: {e}")

    # Speed
    try:
        speed = val_results.speed
        result["inference_ms"] = round(speed.get("inference", 0), 2)
        result["preprocess_ms"] = round(speed.get("preprocess", 0), 2)
        result["postprocess_ms"] = round(speed.get("postprocess", 0), 2)
    except:
        pass

    # Inference time benchmark (single image, averaged)
    val_img_dir = PROJECT_ROOT / "finetune_data" / "images" / "val"
    images = list(val_img_dir.glob("*.jpg"))[:5]  # Use 5 images
    if images:
        times = []
        for img in images:
            t0 = time.time()
            model.predict(str(img), conf=0.15, verbose=False)
            times.append((time.time() - t0) * 1000)  # ms
        result["avg_predict_ms"] = round(sum(times) / len(times), 2)

    result["val_time_sec"] = round(val_time, 1)

    return result


def main():
    print("=" * 70)
    print("  QUANTIZATION BENCHMARK")
    print("=" * 70)

    # Load export results to find all exported model paths
    export_json = RESULTS_DIR / "export_results.json"
    if not export_json.exists():
        print("No export results found. Run 03_export_quantize.py first.")
        return

    with open(export_json) as f:
        exports = json.load(f)

    all_benchmarks = []

    for export in exports:
        if "error" in export:
            continue
        path = export.get("exported_path")
        fmt = export.get("format", "unknown")

        if not path or not os.path.exists(path):
            print(f"  Skipping {fmt}: path not found ({path})")
            continue

        try:
            result = benchmark_model(path, fmt)
            all_benchmarks.append(result)
            print(f"    P={result.get('precision')} R={result.get('recall')} "
                  f"F1={result.get('f1')} mAP50={result.get('mAP50')} "
                  f"Inf={result.get('inference_ms')}ms Size={result.get('size_mb')}MB")
        except Exception as e:
            print(f"  ERROR benchmarking {fmt}: {e}")
            all_benchmarks.append({"format": fmt, "error": str(e)})

    # ─── Save results ────────────────────────────────────
    json_path = RESULTS_DIR / "quantization_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2, default=str)

    csv_path = RESULTS_DIR / "quantization_benchmark.csv"
    if all_benchmarks:
        fieldnames = sorted(set(k for r in all_benchmarks for k in r.keys()))
        priority = ["format", "precision", "recall", "f1", "mAP50", "mAP50_95",
                     "inference_ms", "avg_predict_ms", "size_mb"]
        ordered = [c for c in priority if c in fieldnames]
        ordered += [c for c in fieldnames if c not in ordered]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_benchmarks)

    print(f"\nBenchmark results: {json_path}")
    print(f"Benchmark results: {csv_path}")

    # ─── Summary table ───────────────────────────────────
    print(f"\n{'='*100}")
    print("QUANTIZATION BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"{'Format':<18} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7} {'Inf(ms)':>8} "
          f"{'Predict(ms)':>12} {'Size(MB)':>9}")
    print("-" * 100)

    baseline = None
    for r in all_benchmarks:
        if "error" in r:
            print(f"{r['format']:<18} ERROR: {r['error'][:60]}")
            continue
        if baseline is None:
            baseline = r

        # Size reduction
        size_str = f"{r.get('size_mb', '-')}"
        if baseline and r.get("size_mb") and baseline.get("size_mb"):
            reduction = (1 - r["size_mb"] / baseline["size_mb"]) * 100
            if reduction > 0:
                size_str += f" (-{reduction:.0f}%)"

        print(f"{r['format']:<18} {r.get('precision', '-'):>7} {r.get('recall', '-'):>7} "
              f"{r.get('f1', '-'):>7} {r.get('mAP50', '-'):>7} "
              f"{r.get('inference_ms', '-'):>8} {r.get('avg_predict_ms', '-'):>12} "
              f"{size_str:>9}")
    print(f"{'='*100}")

    # Recommend best for mobile
    valid = [r for r in all_benchmarks
             if "error" not in r and r.get("mAP50") and r.get("size_mb")]
    if valid:
        # Best accuracy
        best_acc = max(valid, key=lambda x: x["mAP50"])
        # Smallest with acceptable accuracy (>80% of best)
        threshold = best_acc["mAP50"] * 0.80
        mobile_candidates = [r for r in valid
                            if r["mAP50"] >= threshold]
        best_mobile = min(mobile_candidates, key=lambda x: x["size_mb"])

        print(f"\nRECOMMENDATIONS:")
        print(f"  Best accuracy:  {best_acc['format']} (mAP50={best_acc['mAP50']})")
        print(f"  Best for mobile: {best_mobile['format']} "
              f"(mAP50={best_mobile['mAP50']}, size={best_mobile['size_mb']}MB)")


if __name__ == "__main__":
    main()
