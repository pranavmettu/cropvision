PYTHON ?= python
DATA_DIR ?= data/raw/plantvillage

.PHONY: setup inspect train evaluate external-validate weather gallery export-onnx benchmark mlflow-train app check health test

setup:
	$(PYTHON) -m pip install -r requirements.txt

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

export-onnx:
	$(PYTHON) -m src.export_onnx

benchmark:
	$(PYTHON) -m src.benchmark_inference

mlflow-train:
	$(PYTHON) -m src.train_cv --data_dir $(DATA_DIR) --epochs 3 --batch_size 16 --use_mlflow

app:
	streamlit run app/streamlit_app.py

health:
	$(PYTHON) scripts/health_check.py

test:
	$(PYTHON) -m unittest discover -s tests

check:
	$(PYTHON) -m compileall src app scripts tests
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) scripts/health_check.py
