# CropVision Disease Model Card

## Model Architecture

Train with `python -m src.train_disease_model`. Supported CPU-friendly architectures are ResNet18, EfficientNet-B0, and MobileNetV3-Small.

## Training Dataset

Expected ImageFolder dataset: `data/processed/cropvision_reference_train`.

## Intended Use

Educational plant health screening demo for visible leaf/crop disease and problem classification.

## Not Intended Use

Not professional crop diagnosis, treatment advice, pesticide guidance, or a replacement for expert agronomic review.

## Metrics

Run `python -m src.evaluate_disease_model --data_dir data/processed/cropvision_reference_train` to populate metrics and reports.

## Known Limitations

- Public plant disease datasets are often controlled and may not match field conditions.
- Unknown plants or diseases can produce uncertain or incorrect predictions.
- Confidence is not agronomic certainty.
- Visual similarity retrieval is supporting evidence, not proof of diagnosis.

## Domain Shift Risks

Models trained on clean leaf images may perform worse on images with cluttered backgrounds, lighting changes, multiple leaves, occlusions, or real field damage.

## Confidence Thresholding

Low-confidence predictions should be surfaced as uncertain instead of forcing a diagnosis.

## Feedback and Retraining

The Streamlit app does not self-train automatically. Uploaded images are saved only after human confirmation and can be used later through `src.retrain_disease_with_feedback`.
