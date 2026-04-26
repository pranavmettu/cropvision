# CropVision

![Project Status](https://img.shields.io/badge/status-portfolio_demo-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-informational)
![ML Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20Streamlit%20%7C%20scikit--learn-green)

CropVision is a multimodal agtech ML portfolio project for plant disease screening. It combines a PyTorch transfer-learning image classifier, Grad-CAM visual explanations, confidence-aware predictions, richer evaluation reports, and an optional weather-based disease/stress risk model using NASA POWER daily weather data.

This tool is for educational ML demonstration only, not professional crop diagnosis.

## Roadmap Progress

[██████████░░░░░░░░░░] 50% complete after core model, Grad-CAM, weather risk, and Streamlit app

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for the full milestone checklist.

## Why It Matters

Plant disease detection is a practical agtech problem where computer vision can help screen crop health faster and more consistently. Real field risk also depends on context, so CropVision demonstrates a transparent multimodal workflow: image evidence from leaf appearance plus weather risk features such as rainfall, humidity, and heat stress.

## Features

- Transfer-learning classifier with ResNet18 or EfficientNet-B0 from `torchvision`
- CPU-friendly training defaults with optional frozen backbone
- ImageFolder dataset support for PlantVillage-style datasets
- Dataset inspection, class distribution reports, and imbalance warnings
- Weighted loss option, early stopping, learning-rate scheduler, and train history CSV
- Evaluation with accuracy, macro F1, weighted F1, top-3 accuracy, per-class metrics, confusion matrices, calibration curve, and misclassified examples
- Confidence thresholding so predictions can be marked uncertain
- Grad-CAM overlays and Grad-CAM gallery generation
- NASA POWER daily weather feature fetching
- Synthetic RandomForest weather-risk model saved with `joblib`
- Streamlit dashboard with upload, demo image support, Grad-CAM, top-k confidence, uncertainty warnings, and optional weather risk
- Health check script, Makefile commands, basic tests, and model card template

## Project Structure

```text
cropvision/
  README.md
  PROJECT_PROGRESS.md
  Makefile
  requirements.txt
  .env.example
  data/
    raw/
    processed/
  models/
  reports/
    figures/
    model_card.md
  sample_images/          # optional local demo images
  scripts/
    health_check.py
  src/
    calibration.py
    config.py
    dataset.py
    evaluate_cv.py
    generate_gradcam_gallery.py
    gradcam.py
    inspect_dataset.py
    multimodal_predict.py
    predict_cv.py
    train_cv.py
    train_weather_model.py
    utils.py
    weather_features.py
  app/
    streamlit_app.py
  tests/
```

## Dataset Instructions

Use PlantVillage or any ImageFolder-style plant disease dataset.

Place images here:

```text
data/raw/plantvillage/
  class_1/
    image1.jpg
  class_2/
    image2.jpg
```

TODO: Download your chosen dataset manually and put class folders under `data/raw/plantvillage/`. The project does not include image data.

Optional demo images for the app can be placed in:

```text
sample_images/
```

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

## Commands

Inspect dataset:

```bash
python -m src.inspect_dataset --data_dir data/raw/plantvillage
```

Train image model:

```bash
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 3
```

CPU-friendly quick smoke run:

```bash
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 1 --max_images_per_class 20 --freeze_backbone
```

Train with imbalance handling:

```bash
python -m src.train_cv --data_dir data/raw/plantvillage --epochs 5 --weighted_loss
```

Evaluate:

```bash
python -m src.evaluate_cv --data_dir data/raw/plantvillage
```

Prediction with confidence threshold:

```bash
python -m src.predict_cv --image_path path/to/leaf.jpg --confidence_threshold 0.6
```

Grad-CAM for one image:

```bash
python -m src.gradcam --image_path path/to/leaf.jpg --confidence_threshold 0.6
```

Grad-CAM gallery:

```bash
python -m src.generate_gradcam_gallery --data_dir data/raw/plantvillage --num_images 12
```

Train weather risk model:

```bash
python -m src.train_weather_model
```

Run Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Health check:

```bash
python scripts/health_check.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Makefile Shortcuts

```bash
make setup
make inspect
make train
make evaluate
make weather
make gallery
make app
make check
```

Set a custom Python executable if needed:

```bash
make check PYTHON=.venv/bin/python
```

## Outputs

Training saves:

- `models/cropvision_cv.pt`
- `models/class_names.json`
- `reports/train_history.csv`
- `reports/figures/loss_curve.png`
- `reports/figures/accuracy_curve.png`
- `reports/figures/class_distribution.png`

Evaluation saves:

- `reports/classification_report.csv`
- `reports/eval_metrics.json`
- `reports/model_card.md`
- `reports/figures/confusion_matrix.png`
- `reports/figures/confusion_matrix_normalized.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/misclassified_examples/`

Grad-CAM gallery saves:

- `reports/figures/gradcam_gallery/`
- `reports/figures/gradcam_gallery/gradcam_gallery.csv`

## Calibration And Uncertainty

CropVision calculates Expected Calibration Error and saves a reliability diagram during evaluation. At prediction time, the confidence threshold can mark an image as `uncertain` instead of forcing a disease label. Low confidence can indicate that the image is out-of-distribution, low quality, ambiguous, or not represented in the training data.

`src.calibration.TemperatureScaler` is included for optional validation-logit temperature scaling experiments.

## Weather Risk Model

The weather model uses a synthetic RandomForest demo trained on simulated labels from rainfall, humidity, wet days, and heat stress. It is designed to show how image predictions can be combined with field context, not to provide agronomic advice.

## Limitations

- PlantVillage images are controlled and may not generalize well to real field images.
- Weather risk model is synthetic/demo unless trained on real disease incidence data.
- Grad-CAM visualizations are interpretability aids, not proof of causal disease symptoms.
- Confidence scores can be miscalibrated, especially after dataset shift.
- This project is not a replacement for expert agronomic diagnosis.

## Future Improvements

- External validation on PlantDoc
- Add object detection or segmentation for lesions
- Train with real field images
- Add persisted temperature scaling and stronger calibration workflows
- Add model monitoring and dataset drift checks
- Deploy with Docker or Hugging Face Spaces

## Resume Bullets

- Built an explainable plant disease classification system using PyTorch transfer learning, Grad-CAM visualizations, and confidence-aware predictions.
- Developed a multimodal agtech ML dashboard combining leaf image classification with weather-derived crop stress risk features.
- Implemented model evaluation tooling including top-k accuracy, macro F1, confusion matrices, calibration analysis, and misclassification galleries.
- Designed a reproducible ML project structure with training, evaluation, reporting, testing, and Streamlit deployment workflows.

## Interview Explanation

“CropVision is a multimodal agtech ML project I built to simulate how computer vision could support crop disease screening. The core model uses transfer learning to classify leaf images, while Grad-CAM explains which visual regions influenced predictions. I added confidence thresholding and calibration checks because in real agricultural settings, a model should know when it is uncertain. I also added a weather-risk component using rainfall, humidity, and temperature features to show how image-based predictions can be combined with field context.”
