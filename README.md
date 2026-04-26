# CropVision: Multimodal Plant Disease Diagnosis

CropVision is a multimodal agtech ML project that uses computer vision to classify plant leaf diseases, Grad-CAM to explain model predictions, and weather-derived features to estimate contextual crop stress risk.

**Live demo:** TODO: Add Hugging Face Spaces URL after deployment.

> This project is for educational ML demonstration only. It is not professional crop diagnosis or agronomic advice.

## One-Sentence Summary

An end-to-end, CPU-friendly ML engineering case study for plant disease screening with PyTorch transfer learning, explainability, uncertainty handling, external validation, weather risk modeling, Streamlit deployment, and inference benchmarking.

## Problem Statement

Plant disease screening is difficult to scale because visual symptoms vary by crop, disease stage, environment, camera quality, and field conditions. A simple image classifier can look impressive on clean datasets, but real-world credibility requires evaluation, uncertainty, explainability, and deployment awareness.

CropVision demonstrates what a stronger portfolio-ready workflow looks like: train a baseline vision model, explain predictions with Grad-CAM, report model quality, flag uncertain predictions, validate on external data, and benchmark inference for deployment.

## Why This Matters For Agtech

Early disease detection can help farmers, agronomists, and researchers prioritize scouting and reduce crop losses. In practice, image symptoms alone are not enough; weather context such as rainfall, humidity, temperature, wet days, and heat stress can affect disease pressure. CropVision shows how leaf image predictions can be combined with environmental signals while clearly communicating limitations.

## Key Features

- PyTorch transfer learning with ResNet18 or EfficientNet-B0
- CPU-friendly defaults with optional frozen backbone training
- ImageFolder support for PlantVillage-style datasets
- Dataset inspection and class distribution reporting
- Training history, early stopping, LR scheduling, and optional weighted loss
- Evaluation with accuracy, macro F1, weighted F1, top-3 accuracy, confusion matrices, calibration curve, and misclassification gallery
- External validation on PlantDoc or any ImageFolder-style dataset with overlapping classes
- Confidence thresholding and uncertainty warnings
- Grad-CAM single-image explanations and gallery generation
- NASA POWER weather feature extraction and synthetic RandomForest risk model
- Optional MLflow experiment tracking
- ONNX export and CPU inference benchmarking
- Streamlit app and Hugging Face Spaces deployment entrypoint
- Health check script, Makefile workflow, tests, and model card template

## Technical Architecture

```text
ImageFolder dataset
  -> dataset inspection
  -> PyTorch transfer-learning training
  -> checkpoint + class_names.json
  -> evaluation reports + calibration + model card
  -> Grad-CAM explanations
  -> Streamlit app

External ImageFolder dataset
  -> overlapping class filter
  -> external validation report

NASA POWER weather data
  -> engineered weather features
  -> synthetic RandomForest risk model
  -> optional multimodal interpretation

Trained PyTorch checkpoint
  -> ONNX export
  -> CPU inference benchmark
```

## Dataset Setup

Use PlantVillage or any ImageFolder-style disease dataset:

```text
data/raw/plantvillage/
  Apple___healthy/
    image1.jpg
  Apple___Apple_scab/
    image2.jpg
```

TODO: Download your dataset manually and place class folders under `data/raw/plantvillage/`. The repo does not include training images.

For external validation, use PlantDoc or another ImageFolder dataset:

```text
data/raw/plantdoc/
  Apple___healthy/
  Apple___Apple_scab/
```

External validation only scores classes whose folder names overlap with `models/class_names.json`; unknown folders are skipped and reported.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Inspect the dataset:

```bash
python -m src.inspect_dataset --data_dir data/raw/plantvillage
```

Train a CPU-friendly model:

```bash
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 3
```

Quick smoke test:

```bash
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 1 --max_images_per_class 20
```

Optional MLflow tracking:

```bash
pip install mlflow
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 3 --use_mlflow
```

MLflow is optional. If it is not installed, training continues normally with a helpful message.

## Evaluation

```bash
python -m src.evaluate_cv --data_dir data/raw/plantvillage
```

Outputs:

- `reports/classification_report.csv`
- `reports/eval_metrics.json`
- `reports/model_card.md`
- `reports/figures/confusion_matrix.png`
- `reports/figures/confusion_matrix_normalized.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/misclassified_examples/`

## External Validation

```bash
python -m src.external_validate --data_dir data/raw/plantdoc
```

Outputs:

- `reports/external_validation_metrics.json`
- `reports/external_validation_report.csv`
- `reports/external_validation_report.md`
- `reports/figures/external_confusion_matrix.png`
- `reports/figures/external_confusion_matrix_normalized.png`

External validation is intentionally strict about domain shift. Clean PlantVillage-style images often differ from real field images in lighting, background complexity, occlusion, blur, disease stage, and camera distance. Weak external validation is useful evidence: it shows where the model is likely to fail outside the original training distribution.

## Calibration And Uncertainty

CropVision computes Expected Calibration Error during evaluation and saves a reliability diagram. During prediction and in the app, a confidence threshold can return `uncertain` rather than forcing a disease label.

```bash
python -m src.predict_cv --image_path path/to/leaf.jpg --confidence_threshold 0.6
```

Low confidence can indicate an out-of-distribution image, poor image quality, ambiguous symptoms, or a disease class not represented in training data.

## Grad-CAM Explainability

Single-image Grad-CAM:

```bash
python -m src.gradcam --image_path path/to/leaf.jpg
```

Gallery:

```bash
python -m src.generate_gradcam_gallery --data_dir data/raw/plantvillage --num_images 12
```

Outputs are saved under `reports/figures/gradcam_gallery/`.

## Weather-Risk Module

Train the synthetic weather risk model:

```bash
python -m src.train_weather_model
```

The weather module can fetch NASA POWER daily data and engineer features such as rainfall, humidity, average temperature, max temperature, heat stress days, and wet days. The included risk model is a synthetic RandomForest demo and should be replaced with real disease incidence labels for practical use.

## ONNX Export And Benchmarking

Export the trained model:

```bash
python -m src.export_onnx
```

Optional ONNX tools:

```bash
pip install onnx onnxruntime
```

Benchmark PyTorch CPU inference and ONNX Runtime if installed:

```bash
python -m src.benchmark_inference --iterations 50
```

Output:

- `reports/inference_benchmark.json`

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- Image upload
- Optional demo images from `sample_images/`
- Top prediction and top-3 probabilities
- Confidence threshold slider
- Uncertainty warnings
- Grad-CAM overlay
- Optional NASA POWER weather risk
- Educational disclaimer

The app launches without requiring external APIs. Weather features are only fetched when the user enables that option.

## Hugging Face Spaces Deployment

Create a new Hugging Face Space:

1. Choose **Streamlit** as the SDK.
2. Use CPU hardware.
3. Upload or push this repo.
4. Hugging Face Spaces will use root `app.py`, `requirements.txt`, and `packages.txt`.
5. For a fully functional image demo, include a trained checkpoint:
   - `models/cropvision_cv.pt`
   - `models/class_names.json`
6. For a lightweight public demo, omit the checkpoint and the app will show a friendly “train the model first” message.

Do not hardcode secrets. No GPU is required.

## Results Table Placeholder

Replace this table after training on your dataset.

| Evaluation setting | Accuracy | Macro F1 | Weighted F1 | Top-3 accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Internal validation | TODO | TODO | TODO | TODO | PlantVillage-style split |
| External validation | TODO | TODO | TODO | TODO | PlantDoc or field-style data |
| PyTorch CPU latency | TODO |  |  |  | See `reports/inference_benchmark.json` |
| ONNX CPU latency | TODO |  |  |  | Optional ONNX Runtime |

## Makefile Workflow

```bash
make setup
make inspect
make train
make evaluate
make external-validate DATA_DIR=data/raw/plantdoc
make export-onnx
make benchmark
make mlflow-train
make app
make check
```

Use the local virtual environment explicitly if needed:

```bash
make check PYTHON=.venv/bin/python
```

## Known Limitations

- PlantVillage images are controlled and may not generalize to field images.
- External validation depends on exact class-name overlap.
- Weather risk is synthetic/demo unless trained with real disease incidence data.
- Grad-CAM is an interpretability aid, not proof of causal disease symptoms.
- Confidence scores can be miscalibrated after dataset shift.
- This project is not a replacement for expert agronomic diagnosis.

## Future Improvements

- Train and validate on larger field-image datasets.
- Add stronger calibration with persisted temperature scaling.
- Add model drift checks for deployed demos.
- Add lesion segmentation or object detection as future work.
- Add Docker deployment after the core ML workflow is stable.

## Resume Bullets

- Built an explainable plant disease classification system using PyTorch transfer learning, Grad-CAM visualizations, and confidence-aware predictions.
- Developed a multimodal agtech ML dashboard combining leaf image classification with weather-derived crop stress risk features.
- Implemented model evaluation tooling including top-k accuracy, macro F1, confusion matrices, calibration analysis, external validation, and misclassification galleries.
- Added deployment-oriented ML engineering features including ONNX export, CPU inference benchmarking, optional MLflow tracking, health checks, tests, and a Streamlit/Hugging Face Spaces workflow.

## Interview Explanation

CropVision is a multimodal agtech ML project that uses computer vision to classify plant leaf diseases, Grad-CAM to explain model predictions, and weather-derived features to estimate contextual crop stress risk. I added calibration, uncertainty handling, external validation, and inference benchmarking to make the project closer to a real ML engineering workflow rather than a simple image classifier.
