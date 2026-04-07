# Plant Grid - Spatial Plant Counting Prototype

This repository contains a notebook-first prototype for counting plants from mobile field video using spatial event logic.

## Main Goal

Count each unique plant position once while reducing misses and duplicate counts in dense agricultural rows.

## Key Files

- `Untitled25.ipynb`: primary notebook pipeline.
- `_run_pipeline.py`: script-style pipeline variant.
- `_finetune_prep.py`: pseudo-label data preparation for fine-tuning.
- `_show_audit.py`: quick audit/missed-track inspection utility.
- `project_progress_report.md`: full detailed progress report.
- `project_progress_llm_training_report.md`: compact LLM-handoff report.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook `Untitled25.ipynb` in order:
- Cell 1: dependencies
- Cell 2: input selection
- Cell 3: counting pipeline
- Cell 4: summary and output preview

## Notes

- Large binaries (models/videos/training outputs) are intentionally ignored via `.gitignore`.
- For production quality, fine-tune on your real crop dataset and camera angle.
