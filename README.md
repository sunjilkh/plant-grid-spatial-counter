# Plant Grid Spatial Counter

Spatial plant counting prototype for row-based agriculture video.

## What this repo contains

- Notebook pipeline: `Untitled25.ipynb`
- Local web app: `app.py`
- Fine-tuning dataset: `finetune_data/`
- Fine-tuning dataset generator: `_finetune_prep.py`
- Project reports:
  - `project_progress_report.md`
  - `project_progress_llm_training_report.md`

## Core objective

Count plants by spatial position (one position = one plant), not appearance-only identity.

## Quick start

1. Create/activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run web app:

```powershell
python app.py
```

4. Or run the notebook pipeline in `Untitled25.ipynb`.

## Fine-tuning data

Dataset is in YOLO format under `finetune_data/`.

Train example:

```powershell
yolo train model=yolov8m.pt data=finetune_data/data.yaml epochs=30 imgsz=640 batch=4 name=plant_finetune
```

## Notes

- Generated runtime artifacts are ignored (`output_*.mp4`, audit CSVs, summaries).
- Model weight files (`*.pt`) are ignored; add your trained weights locally (for example `plant_finetuned.pt`) when running inference.
