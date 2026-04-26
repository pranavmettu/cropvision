# CropVision Portfolio Roadmap

[██████████░░░░░░░░░░] 50% complete after core model, Grad-CAM, weather risk, and Streamlit app

## Next Feature Milestones

[████████████░░░░░░░░] 60% - Better evaluation and reporting

- [x] Add richer model evaluation outputs.
- [x] Add per-class precision, recall, F1, and support.
- [x] Save classification report as `reports/classification_report.csv`.
- [x] Save confusion matrix as `reports/figures/confusion_matrix.png`.
- [x] Save normalized confusion matrix as `reports/figures/confusion_matrix_normalized.png`.
- [x] Add top-k accuracy, especially top-3 accuracy.
- [x] Add misclassified examples gallery.

[██████████████░░░░░░] 70% - Robustness and confidence calibration

- [x] Add confidence thresholding so the app can say "uncertain" instead of forcing a prediction.
- [x] Add calibration metrics such as Expected Calibration Error.
- [x] Add optional temperature scaling utility for validation predictions.
- [x] Save calibration plot to `reports/figures/calibration_curve.png`.
- [x] Add a command-line argument for confidence threshold.
- [x] Show top prediction, confidence, and uncertainty warning in the app.

[████████████████░░░░] 80% - Dataset quality and training improvements

- [x] Add class distribution report.
- [x] Save class distribution plot to `reports/figures/class_distribution.png`.
- [x] Add weighted loss for imbalanced classes.
- [x] Add early stopping.
- [x] Add learning rate scheduler.
- [x] Add optional frozen-backbone training for faster CPU testing.
- [x] Add train history CSV saved to `reports/train_history.csv`.
- [x] Add train/validation accuracy and loss curves.

[██████████████████░░] 90% - Portfolio-grade explainability and demo outputs

- [x] Improve Grad-CAM reliability and uncertainty metadata.
- [x] Add a script that generates a Grad-CAM gallery for sample images.
- [x] Save outputs to `reports/figures/gradcam_gallery/`.
- [x] Add model card at `reports/model_card.md`.
- [x] Add a demo mode in Streamlit that works with sample images if available.
- [x] Add clear limitations in app and README.
- [x] Add "not professional crop diagnosis" warning in the app.

[████████████████████] 100% - Engineering polish

- [x] Add tests for utility functions where practical.
- [x] Add a Makefile with common commands.
- [x] Add a clean `requirements.txt`.
- [x] Add `.env.example` for API-related settings.
- [x] Add a repo health check script.
- [x] Add README badges as plain markdown placeholders.
- [x] Add final resume bullets and interview explanation section.

## Current Status

CropVision now has a complete portfolio-ready ML workflow: dataset inspection, CPU-friendly training, richer evaluation, calibration analysis, Grad-CAM galleries, Streamlit demo UX, health checks, and basic tests.
