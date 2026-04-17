# PlantVision AI: Professional Spatial Plant Counting

An end-to-end, research-grade system for detecting and counting unique plants in agricultural environments. This project combines state-of-the-art object detection (YOLOv8) with spatial tracking to ensure each plant is counted exactly once, even in complex video sequences.

## 🚀 Key Features

*   **Custom YOLOv8 Training**: Fine-tuned on agricultural data for high-precision plant identification.
*   **Spatial Counting & Tracking**: Uses BoT-SORT/ByteTrack to maintain unique identities for plants across frames.
*   **Research Ablation Study**: Systematic comparison of model variants (Nano, Small, Medium, Large) across accuracy, speed, and size.
*   **Mobile Optimized**: Support for ONNX and TFLite quantization (FP16/INT8) for edge device deployment.
*   **Production Flask App**: Premium dark-themed web interface for easy image/video analysis.
*   **Auditability**: Detailed CSV logs of every detection event for verification and diagnostics.

## 📁 Project Structure

*   `research/`: Scripts for training, evaluation, validation, and benchmarking.
*   `flask_app/`: Production-ready web server and premium UI.
*   `finetune_data/`: Dataset configuration and training labels.
*   `_finetune_prep.py`: Preprocessing utility for data preparation.

## 🛠 Usage

### Training & Research
Run the ablation study to find the best model for your needs:
```bash
python research/01_train_ablation.py
python research/02_evaluate_models.py
```

### Quantization
Export the best model for mobile/edge use:
```bash
python research/03_export_quantize.py
```

### Deployment
Launch the interactive web portal:
```bash
python flask_app/app.py
```

## 📊 Methodology
This project follows a rigorous research approach, documenting Precision, Recall, F1, mAP, and inference latency across all model variants and quantization levels.
