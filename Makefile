PYTHON ?= python
DATA_DIR ?= data/raw/plantvillage

.PHONY: setup datasets-list import-plantvillage import-plantdoc build-reference-data train-reference inspect train evaluate external-validate weather gallery retrieval build-reference-index export-onnx benchmark mlflow-train retrain-feedback advanced-check app check health test

setup:
	$(PYTHON) -m pip install -r requirements.txt

datasets-list:
	$(PYTHON) -m src.dataset_registry --list

import-plantvillage:
	$(PYTHON) -m src.dataset_manager --import-local --dataset plantvillage --source_dir data/raw/plantvillage --output_dir data/processed/reference_datasets/plantvillage

import-plantdoc:
	$(PYTHON) -m src.dataset_manager --import-local --dataset plantdoc --source_dir data/raw/plantdoc --output_dir data/processed/reference_datasets/plantdoc

build-reference-data:
	$(PYTHON) -m src.build_reference_dataset --datasets plantvillage,new_plant_diseases_kaggle --source_root data/processed/reference_datasets --output_dir data/processed/cropvision_reference_train --normalize_labels --max_images_per_class 1000 --seed 42

train-reference:
	$(PYTHON) -m src.train_cv --data_dir data/processed/cropvision_reference_train --model_name efficientnet_b0 --epochs 8 --batch_size 16 --freeze_backbone --weighted_loss --label_smoothing 0.05 --model_version_name reference_v1

inspect:
	$(PYTHON) -m src.inspect_dataset --data_dir $(DATA_DIR)

train:
	$(PYTHON) -m src.train_cv --data_dir $(DATA_DIR) --epochs 3 --batch_size 16

evaluate:
	$(PYTHON) -m src.evaluate_cv --data_dir $(DATA_DIR)

external-validate:
	$(PYTHON) -m src.external_validate --data_dir $(DATA_DIR)

weather:
	$(PYTHON) -m src.train_weather_model

gallery:
	$(PYTHON) -m src.generate_gradcam_gallery --data_dir $(DATA_DIR) --num_images 12

retrieval:
	$(PYTHON) -m src.image_retrieval --data_dir data/processed/plantvillage_sample --build_index

build-reference-index:
	$(PYTHON) -m src.reference_retrieval --build_index --data_dir data/processed/cropvision_reference_train --output_dir models/reference_index

export-onnx:
	$(PYTHON) -m src.export_onnx

benchmark:
	$(PYTHON) -m src.benchmark_inference

mlflow-train:
	$(PYTHON) -m src.train_cv --data_dir $(DATA_DIR) --epochs 3 --batch_size 16 --use_mlflow

retrain-feedback:
	$(PYTHON) -m src.retrain_with_feedback --base_data_dir data/processed/cropvision_reference_train --feedback_dir data/user_feedback/verified --model_version_name reference_plus_feedback_v1

app:
	streamlit run app/streamlit_app.py

health:
	$(PYTHON) scripts/health_check.py

advanced-check:
	$(PYTHON) scripts/run_advanced_demo_check.py

test:
	$(PYTHON) -m unittest discover -s tests

check:
	$(PYTHON) -m compileall src app scripts tests
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) scripts/health_check.py
