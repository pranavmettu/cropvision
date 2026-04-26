# CropVision Dataset Card

## Dataset Sources

- PlantVillage: controlled leaf images, intended as core reference training data.
- New Plant Diseases Dataset from Kaggle: larger controlled RGB leaf dataset, optional core training data.
- PlantDoc: field-style images, intended mainly for external validation.

## Dataset Construction

Use `src.dataset_manager` to import local downloads and `src.build_reference_dataset` to merge and normalize labels into `data/processed/cropvision_reference_train/`.

## Known Biases And Limitations

- PlantVillage-style images are controlled and may overestimate field performance.
- PlantDoc-style images include domain shift from backgrounds, lighting, occlusion, and camera distance.
- Class names and disease taxonomies differ across sources and require normalization.
- Class imbalance should be checked before training.

## License And Terms Reminder

Each source dataset has its own license and terms. Do not redistribute image data unless the original license permits it.
