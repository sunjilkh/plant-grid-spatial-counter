# Plant Grid Project Report

## Purpose

This report documents the full progress of the plant counting project in this workspace. It is intended both as a technical project summary and as a handoff document for a stronger local LLM or future engineering work.

The core objective is not generic object detection. The goal is a spatial plant identity system for agriculture, where each physical plant position should be counted once, even when plants are visually similar, partially visible, occluded, or changing over time.

## Final Project Aim

The target problem is the "10 images = 1 plant" challenge for large-scale agriculture. The system must support mobile capture across dense crop rows and eventually scale to 1,000 to 10,000 plants.

The intended identity rule is spatial, not appearance-based:

- A plant belongs to a fixed physical position.
- If that position is observed repeatedly across frames, it should map to the same plant identity.
- The count must increase only once per unique plant position.
- No clearly visible plant should be missed if the detector and geometry are working correctly.
- No plant should be counted more than once.

## Research and Engineering Summary

The project evolved from a simple video demo into a more detailed spatial counting prototype.

The reasoning behind the work was:

- Plants of the same species are visually similar, so Re-ID is not reliable as the core identity source.
- As the camera moves, plant positions shift in image space, so pure box overlap logic is insufficient.
- Dense rows and occlusion make simple tracker assumptions fragile.
- The system therefore needs spatial event logic, a detection layer with better recall, and auditability for every count event.

## What Was Built

The notebook [Untitled25.ipynb](Untitled25.ipynb) became the main implementation artifact. It now includes:

- Video loading and environment setup.
- Automated input selection for raw source video.
- YOLO-based plant detection.
- BoT-SORT tracking.
- Spatial lane gating for left and right plant rows.
- Crossing-based event counting.
- Recovery logic for late-created tracks and stable appearances.
- Duplicate-event suppression using time and spatial proximity.
- Video export and summary export.
- Event-level audit logging.

Supporting artifacts created during debugging and validation include:

- [count_summary.json](count_summary.json)
- [event_audit.csv](event_audit.csv)
- [missed_tracks.csv](missed_tracks.csv)
- [output_colab.mp4](output_colab.mp4)

## Implementation Attempts and Iterations

### 1. Initial naive tripwire logic

The first approach used a center-line or small-zone trigger with a bottom-center anchor point from each detection box.

What it did well:

- It proved the pipeline could detect plants and draw meaningful overlays.
- It made the spatial-counting concept visible in the video.

Why it failed:

- A plant could jump over the trigger zone between frames.
- The line geometry was too fragile for walking motion and perspective change.
- IDs could be lost or reassigned near occlusion.
- Several clearly visible plants were never counted.

### 2. Occupancy-based lane gating

The next attempt moved to lane polygons and occupancy logic, using a one-position-one-plant idea.

What it improved:

- It moved the system away from a single narrow line.
- It gave a more agriculture-like idea of row regions.

Why it failed:

- In a dense crop row, the lane is often never truly empty.
- Plants overlap temporally and spatially, so an empty-to-occupied state machine undercounts badly.
- This logic is too strict for a continuous row with a moving camera.

### 3. Vector crossing and ROI debounce

The pipeline was then upgraded to line-crossing using segment intersection and a center ROI debounce.

What it improved:

- Crossing detection became more robust to sudden motion.
- The system stopped relying on a plant being inside one exact pixel window.
- The count became non-zero and more stable.

Why it still failed:

- Some detections appeared late, after the scan condition was already passed.
- Partial visibility made some boxes unstable.
- Lane occupancy was still too strict for dense rows.

### 4. BoT-SORT with recovery logic

The notebook then moved to BoT-SORT tracking and added recovery events.

Recovery events included:

- Late-created tracks that first appear below the scan line.
- Stable appearance from above the scan line even if crossing is not clean.
- Time and position-based duplicate suppression.

What it improved:

- Tracker continuity became more reliable across short occlusions.
- The system captured more visible plants than earlier versions.
- The count rose into a meaningful range.

Remaining issue:

- Even with recovery logic, the detector remained the main bottleneck.
- Misses still occurred for partially visible plants or unstable boxes.

### 5. Detector benchmarking

The detector model was benchmarked on sampled frames.

Observed result trend:

- YOLOv8n produced fewer detections than larger models.
- YOLOv8s improved recall.
- YOLOv8m gave the strongest detection recall in the available tests.

This directly improved the final count and was one of the most important changes in the project.

## Validation and Test Results

### Video tested

- [test_video--big.mp4](test_video--big.mp4)

### Final model and logic configuration

- Detector: YOLOv8m
- Tracker: BoT-SORT
- Target class: COCO class 58, potted plant
- Scan line and lane gating were calibrated for the test video
- Duplicate suppression used a time and distance buffer
- Recovery events were enabled for difficult tracks

### Final reported result

From [count_summary.json](count_summary.json):

- Left count: 8
- Right count: 8
- Registered count: 16
- Event counts by reason:
  - cross: 10
  - late: 0
  - appear: 6
- Missed track count: 14
- Missed crossed-band count: 0

### Meaning of the result

This means the final system was able to count 16 plants in the tested clip, and the counting was no longer stuck near zero. It also shows that:

- The event logic can produce stable count increments.
- The detector is still missing a meaningful subset of visible plants.
- Many missed tracks did not cross the counting band, which suggests geometry and track timing are still part of the problem, not just the count rule itself.

## Audit Evidence

The event audit file [event_audit.csv](event_audit.csv) records each successful count with:

- frame
- lane
- track id
- anchor position
- event reason
- bounding box coordinates
- confidence
- count at event

This is important because it separates the count output from the reason behind the count, which makes debugging much easier.

The miss analysis file [missed_tracks.csv](missed_tracks.csv) shows tracks that were seen but never counted. The important pattern in the current miss set is:

- Some tracks were visible for multiple frames but still never crossed the counting condition.
- Some tracks were short-lived or weakly visible.
- Some tracks were on the correct lane but still not captured by the event logic.

## Main Failures Observed

### 1. Detector recall is still incomplete

The biggest remaining limitation is that the detector does not see every plant clearly enough, especially with partial visibility and crop clutter.

This matters because event logic cannot recover plants that are never detected consistently.

### 2. Geometry is still an approximation

The current lane and scan geometry works as a prototype, but it is still image-space geometry, not physical-space geometry.

That means it can drift when:

- the camera angle changes
- walking speed changes
- rows are not centered
- the path is not straight

### 3. Continuous rows are harder than isolated objects

A dense crop row does not behave like a classic object counting scene.

The same plant can be:

- partially hidden
- detected only for a few frames
- briefly merged with nearby foliage
- re-detected under a new track id

### 4. COCO class mismatch is a structural weakness

Using COCO potted plant as a proxy helped the demo, but it is not a production crop model.

For turmeric, paddy, or similar field crops, the detector should be fine-tuned on the actual crop type and camera angle.

## Target Achieved

The project did achieve several important milestones:

- The notebook runs end-to-end.
- The system counts plants spatially rather than by appearance alone.
- The output is auditable through a CSV event log.
- The final pipeline produces a non-zero, meaningful count.
- The final count reached 16 on the validation clip.
- The model comparison showed that larger detectors improve recall.
- The project now has a practical base for future training and calibration.

## Still Not Fully Achieved

The following are still not fully solved:

- Zero-miss counting of all clearly visible plants.
- Strong duplicate-proof identity over long occlusions.
- Physical coordinate mapping from image space to field space.
- Production-level performance on real crop species.
- Guaranteed robustness across all mobile capture paths.

## Most Important Insight From the Work So Far

The main insight is that this project is not just a counting problem. It is a data association problem under spatial uncertainty.

The count accuracy depends on three layers working together:

1. Detector recall
2. Track continuity
3. Spatial event logic

If any one of those three layers fails, the final count can still be wrong.

Right now, the main bottleneck is detector quality and scene-specific calibration, not the existence of a counting rule.

## Best Next Steps

### Immediate next step

Replace the proxy detector with a fine-tuned crop detector trained on your own field data.

### Next engineering step

Add event-level CSV logging for every candidate detection, not only final counts, so missed plants can be inspected frame by frame.

### Next geometry step

Calibrate the lane polygons and scan line from the actual camera path or a clicked frame.

### Next production step

Map image coordinates to physical row coordinates using homography or a similar spatial transform.

### Next validation step

Run a repeatable benchmark on several clips and track:

- miss rate
- duplicate rate
- count stability across repeated runs
- sensitivity to speed and angle

## Recommended Handoff Package for a Stronger Local LLM

If this report is used to train or guide another local LLM, the best input package should include:

- This report.
- The latest notebook [Untitled25.ipynb](Untitled25.ipynb).
- [count_summary.json](count_summary.json).
- [event_audit.csv](event_audit.csv).
- [missed_tracks.csv](missed_tracks.csv).
- Example output frames or the rendered video [output_colab.mp4](output_colab.mp4).

The model should be asked to focus on:

- reducing misses for partially visible plants
- reducing duplicate events under occlusion
- improving lane geometry
- turning the current prototype into a field-grade spatial counting pipeline

## Final Assessment

The project is no longer a concept-only prototype. It is now a working spatial counting pipeline with documented limitations, measurable output, and a clear path forward.

The current implementation demonstrates the right architecture direction, but it is still constrained by proxy detection quality and approximate image-space geometry. The next real leap will come from crop-specific fine-tuning and physical-space calibration.
