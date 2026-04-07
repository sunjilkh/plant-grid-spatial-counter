# Plant Grid Project: LLM Training Report

## Role of This Document

This is the compact handoff version of the plant-counting project report. It is written for a stronger local LLM that needs to continue the work, not restart it.

Use this as the project state summary, failure-analysis brief, and next-step directive.

## Core Problem

Build a spatial plant counter for mobile agriculture capture where:

- one physical plant position equals one plant identity
- each unique plant is counted once and only once
- clearly visible plants should not be missed
- repeated visual similarity must not break identity
- the system must survive partial visibility, occlusion, camera motion, and dense crop rows

The intended identity source is spatial position and event logic, not appearance-based Re-ID.

## Current Technical Strategy

The current notebook implements a practical prototype with:

- YOLO-based plant detection
- BoT-SORT tracking
- lane-based spatial gating
- scan-line crossing events
- late-entry recovery for tracks that appear after the crossing moment
- appearance recovery for stable tracks that do not cross cleanly
- duplicate-event suppression using frame and position distance
- event audit logging and missed-track logging

## What Was Tested

Test video:

- [test_video--big.mp4](test_video--big.mp4)

Detection benchmark trend:

- YOLOv8n under-detected plants
- YOLOv8s improved recall
- YOLOv8m gave the strongest recall in the available tests

This detector comparison was one of the most important findings.

## Final Measured Result

Latest validated run from [count_summary.json](count_summary.json):

- model: yolov8m.pt
- target class: 58, potted plant
- left count: 8
- right count: 8
- total registered count: 16
- event reasons:
  - cross: 10
  - late: 0
  - appear: 6
- missed track count: 14
- missed crossed-band count: 0

The output video is [output_colab.mp4](output_colab.mp4).

## What This Means

### Achieved

- The pipeline now runs end-to-end.
- Counting is no longer stuck near zero.
- The system can produce auditable count events.
- The count is spatially driven rather than purely appearance-driven.
- Larger detector models improve recall materially.

### Still failing

- Some clearly visible plants are still missed.
- The COCO potted-plant class is only a proxy and is not a field crop model.
- Image-space lane geometry is still approximate.
- Dense rows and partial visibility still cause recall gaps.
- The system is not yet production-grade for guaranteed no-miss/no-double counting.

## Main Failure Analysis

The remaining errors are not caused by one issue alone. They come from three stacked weaknesses:

1. Detector recall is still incomplete.
2. Some tracks appear late or fragment under occlusion.
3. The current geometry is still an approximation of real field position, not true physical mapping.

The dominant bottleneck is detector quality and scene calibration, not only counting logic.

## Best Next Improvements

### Highest priority

1. Fine-tune on the real crop and camera angle.
2. Add event-level audit CSV for every candidate count and miss.
3. Calibrate lane geometry from the actual field path.
4. Move from approximate image-space rules toward physical coordinate mapping.

### Validation targets

1. Reduce miss rate on clearly visible plants.
2. Keep duplicate rate near zero.
3. Make repeated runs on the same clip stable.
4. Test across different walking speeds, angles, and lighting.

## Prompt for the Next LLM

Continue this project from the current notebook state.

You are improving a spatial plant counting system for agriculture. Do not redesign the project from scratch.

Your job is to:

- keep the one-position-one-plant principle
- reduce missed plants, especially partially visible ones
- prevent duplicate counts under occlusion and track switching
- improve geometry calibration
- prepare the pipeline for crop-specific fine-tuning

Use this project state as the baseline:

- notebook: [Untitled25.ipynb](Untitled25.ipynb)
- summary: [count_summary.json](count_summary.json)
- event audit: [event_audit.csv](event_audit.csv)
- missed-track log: [missed_tracks.csv](missed_tracks.csv)
- output video: [output_colab.mp4](output_colab.mp4)

Prioritize changes in this order:

1. detector recall
2. event logic robustness
3. geometry calibration
4. physical coordinate mapping
5. auditability and validation metrics

## Short Conclusion

The project is working, but still limited by detector recall and proxy-class mismatch. The current notebook is a valid spatial-counting prototype, not the final field-grade system. The next meaningful gain comes from a crop-specific detector and calibrated physical geometry.
