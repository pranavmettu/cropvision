PYTHON ?= python
DATA_DIR ?= data/raw/plantvillage

.PHONY: setup inspect train evaluate weather gallery app check health test

setup:
	$(PYTHON) -m pip install -r requirements.txt

inspect:
	$(PYTHON) -m src.inspect_dataset --data_dir $(DATA_DIR)

train:
	$(PYTHON) -m src.train_cv --data_dir $(DATA_DIR) --epochs 3 --batch_size 16

evaluate:
	$(PYTHON) -m src.evaluate_cv --data_dir $(DATA_DIR)

weather:
	$(PYTHON) -m src.train_weather_model

gallery:
	$(PYTHON) -m src.generate_gradcam_gallery --data_dir $(DATA_DIR) --num_images 12

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
