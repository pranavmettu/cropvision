# CropVision Model Card

## Model Name

CropVision plant disease image classifier

## Intended Use

Educational ML portfolio demonstration for plant leaf image classification, Grad-CAM explainability, and confidence-aware predictions.

## Not Intended Use

This model is not intended for professional crop diagnosis, agronomic decisions, pesticide decisions, or safety-critical agricultural recommendations.

## Dataset

Expected dataset format: ImageFolder under `data/raw/plantvillage/`.

TODO: Replace this template with the exact dataset source, license, class list, and collection conditions after training.

## Architecture

Supported architectures: ResNet18 and EfficientNet-B0 through `torchvision`.

## Training Setup

CPU-friendly transfer learning with 224x224 images, ImageNet normalization, optional frozen backbone, optional weighted loss, early stopping, and learning-rate scheduling.

## Metrics

Run `python -m src.evaluate_cv --data_dir data/raw/plantvillage` to populate accuracy, macro F1, weighted F1, top-3 accuracy, confusion matrices, calibration curve, and misclassified examples.

## Known Limitations

- PlantVillage-style images are controlled and may not generalize to field images.
- Weather risk is synthetic/demo unless trained on real disease incidence labels.
- Confidence and Grad-CAM are aids for interpretation, not proof of diagnosis.

## Ethical and Practical Risks

Incorrect predictions could lead to inappropriate crop management if misused. Expert review is required for real agricultural decisions.

## Future Improvements

- External validation on PlantDoc.
- Field-image training and calibration.
- Lesion segmentation or object detection.
- Deployment monitoring and drift checks.
