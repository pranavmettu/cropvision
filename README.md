# CropVision: Reference-Data Plant Health Recognition

CropVision is an ML engineering case study for plant health recognition. It uses public plant disease image datasets as reference data for training, evaluation, Grad-CAM explainability, weather-risk context, visual retrieval, and explicit human-verified feedback retraining.

> Educational ML demo only. Not professional crop diagnosis or treatment advice.

## What “Database Access” Means

CropVision uses public plant disease image datasets as reference data for training and retrieval. The model does not access a database magically at prediction time. Instead, dataset ingestion scripts download or import plant disease datasets, normalize labels, build a combined ImageFolder dataset, train a disease classifier, and create a reference image retrieval index. During prediction, the app uses the trained model plus visually similar examples from the reference database. Uploaded user images can be saved only with confirmation and used later for explicit retraining.

## Supported Reference Image Datasets

- **PlantVillage**: controlled leaf images, about 54k images and 38 crop/disease classes. Best as core training data.
- **New Plant Diseases Dataset from Kaggle**: larger controlled RGB leaf dataset, about 87k images and 38 classes. Best as larger reference training data if available.
- **PlantDoc**: real-world field-style images, about 2,598 images. Best for external validation/domain-shift checks.
- **Future species datasets**: PlantCLEF or iNaturalist-style data can support species ID, but they are not required for disease training.

## Dataset Setup

List supported datasets:

```bash
python -m src.dataset_registry --list
```

Import PlantVillage from local folder:

```bash
python -m src.dataset_manager --import-local --dataset plantvillage --source_dir data/raw/plantvillage --output_dir data/processed/reference_datasets/plantvillage
```

Import PlantDoc from local folder:

```bash
python -m src.dataset_manager --import-local --dataset plantdoc --source_dir data/raw/plantdoc --output_dir data/processed/reference_datasets/plantdoc
```

Download Kaggle New Plant Diseases Dataset:

```bash
python -m src.dataset_manager --download-kaggle --dataset new_plant_diseases_kaggle --output_dir data/raw/kaggle_new_plant_diseases
```

Kaggle setup:

1. `pip install kaggle`
2. Create a Kaggle API token from your Kaggle account settings.
3. Place `kaggle.json` in `~/.kaggle/`.
4. Rerun the download command.

Do not commit Kaggle credentials.

## Label Normalization

CropVision normalizes messy labels into:

- `plant_species`
- `disease_name`
- `health_status`
- `normalized_class`
- `broad_problem_category`

Example:

```text
Tomato___Early_blight -> tomato__early_blight
Corn_(maize)___Common_rust_ -> corn_maize__common_rust
```

Label maps are saved to:

- `models/label_map.json`
- `reports/label_normalization_report.csv`

## Build Combined Reference Dataset

```bash
python -m src.build_reference_dataset --datasets plantvillage,new_plant_diseases_kaggle --source_root data/processed/reference_datasets --output_dir data/processed/cropvision_reference_train --normalize_labels --max_images_per_class 1000 --seed 42
```

Outputs:

- `reports/reference_dataset_report.json`
- `reports/reference_dataset_class_distribution.csv`
- `reports/figures/reference_dataset_class_distribution.png`

## Train Reference Disease Model

```bash
python -m src.train_cv --data_dir data/processed/cropvision_reference_train --model_name efficientnet_b0 --epochs 8 --batch_size 16 --freeze_backbone --weighted_loss --label_smoothing 0.05 --model_version_name reference_v1
```

Versioned outputs:

- `models/versions/reference_v1/cropvision_cv.pt`
- `models/versions/reference_v1/class_names.json`
- `models/versions/reference_v1/label_map.json`
- `models/versions/reference_v1/training_config.json`
- `models/versions/reference_v1/metrics.json`

Latest outputs:

- `models/cropvision_cv.pt`
- `models/class_names.json`

## Evaluation

```bash
python -m src.evaluate_cv --data_dir data/processed/cropvision_reference_train
```

## External Validation

External validate on PlantDoc:

```bash
python -m src.external_validate --data_dir data/processed/reference_datasets/plantdoc --model_version reference_v1
```

External validation reports exact class matching, normalized class matching, and broad problem-category matching. This is important because PlantDoc-style field images differ from clean controlled training images.

## Build Reference Retrieval Index

```bash
python -m src.reference_retrieval --build_index --data_dir data/processed/cropvision_reference_train --output_dir models/reference_index
```

Outputs:

- `models/reference_index/embeddings.npz`
- `models/reference_index/index.joblib`
- `models/reference_index/metadata.csv`

If FAISS is installed, CropVision can use it. Otherwise it uses `sklearn.neighbors.NearestNeighbors`.

## Run App

```bash
streamlit run app/streamlit_app.py
```

The app shows:

- Current trained model version
- Number of trained disease classes
- Whether a reference retrieval index exists
- Predicted disease class
- Normalized plant species and disease name
- Broad problem category
- Confidence and top-3 predictions
- Grad-CAM
- Reference examples from training database
- Optional weather risk
- Verified feedback form

Warning: the model can only recognize diseases represented in its training database. Unknown plants or diseases may produce uncertain predictions.

## Use Uploaded Images As Verified Feedback

Uploaded user images are never used for automatic training. If a human confirms or corrects the label in the app, the image is saved under:

```text
data/user_feedback/verified/{normalized_class}/
```

Metadata includes:

- original prediction
- normalized label
- image characteristics
- reference retrieval matches
- model version

## Retrain With Feedback

```bash
python -m src.retrain_with_feedback --base_data_dir data/processed/cropvision_reference_train --feedback_dir data/user_feedback/verified --model_version_name reference_plus_feedback_v1
```

Why not blind self-training? Because model predictions can be wrong, especially for unknown plants, new diseases, low-quality images, or domain-shifted field conditions. CropVision only uses human-verified images for feedback retraining.

## Makefile Commands

```bash
make datasets-list
make import-plantvillage
make import-plantdoc
make build-reference-data
make train-reference
make external-validate DATA_DIR=data/processed/reference_datasets/plantdoc
make build-reference-index
make app
make retrain-feedback
make check
```

Use `.venv` explicitly if needed:

```bash
make check PYTHON=.venv/bin/python
```

## Limitations

- Public datasets have licensing and redistribution constraints.
- PlantVillage and Kaggle images are controlled and may not generalize to field images.
- PlantDoc is useful for external validation but small.
- Label normalization is heuristic and should be reviewed for high-stakes use.
- Retrieval shows visual similarity, not proof of diagnosis.
- No automatic treatment recommendations are provided.

## Resume Bullets

- Built a reference-data plant disease recognition system using public plant disease datasets, PyTorch transfer learning, label normalization, Grad-CAM, and visual retrieval.
- Designed dataset ingestion pipelines for PlantVillage, Kaggle New Plant Diseases, and PlantDoc with import reports, corrupt-image handling, duplicate detection, and normalized ImageFolder outputs.
- Implemented model versioning, external validation, reference image retrieval, and human-verified feedback retraining workflows.
- Developed a Streamlit app that separates trained model predictions from reference examples and avoids blind self-training from user uploads.

## Interview Explanation

CropVision is a plant health recognition project that treats public plant disease datasets as explicit reference data. I built ingestion scripts to import PlantVillage, Kaggle plant disease data, and PlantDoc, normalize labels across datasets, build a combined reference training set, train a versioned disease classifier, and create a visual retrieval index. During prediction, the app uses the trained model and visually similar reference examples, but it does not magically query a database or train on its own predictions. User uploads can only become training data after human confirmation, which makes the workflow safer and more realistic.
