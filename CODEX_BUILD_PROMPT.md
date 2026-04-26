# Prompt for Codex

Build a full ML portfolio project called CropVision: Multimodal Plant Disease Diagnosis Using Computer Vision and Weather-Based Risk Modeling.

Goal:
Create a complete, runnable Python project that lets a user upload a crop/leaf image, predicts the plant disease class using computer vision, explains the prediction with Grad-CAM, and optionally adds a weather-based disease/stress risk score using NASA POWER weather data.

Use the existing folder structure. Make the project easy to run locally in VS Code.

Important:
- Build the codebase, not just a notebook.
- Use clean, modular Python files.
- Include comments where helpful.
- Include a professional README.
- Include requirements.txt.
- Include example commands for training, evaluation, and running the app.
- Make sure all paths work from the project root.
- Use PyTorch for the computer vision model.
- Use Streamlit for the app.
- Use scikit-learn/XGBoost or RandomForest for the simple weather-risk model.
- Do not assume I have a GPU.
- Make the default training setup small enough to test quickly on CPU.
- Add TODO comments where I need to download or place datasets manually.

Project structure:
```text
cropvision/
  README.md
  requirements.txt
  .gitignore
  CODEX_BUILD_PROMPT.md
  data/
    raw/
    processed/
  models/
  reports/
    figures/
  notebooks/
  src/
    __init__.py
    config.py
    dataset.py
    train_cv.py
    evaluate_cv.py
    predict_cv.py
    gradcam.py
    weather_features.py
    train_weather_model.py
    multimodal_predict.py
    utils.py
  app/
    streamlit_app.py
```

Core computer vision requirements:
1. Create a PyTorch image classification pipeline using transfer learning.
2. Use EfficientNet-B0 or ResNet18 from torchvision.
3. Dataset should assume an ImageFolder format:
   data/raw/plantvillage/
     class_1/
       image1.jpg
     class_2/
       image2.jpg
4. Implement train/validation split.
5. Include image transforms:
   - Resize to 224x224
   - RandomHorizontalFlip
   - RandomRotation
   - ColorJitter
   - Normalize using ImageNet mean/std
6. Save the best model checkpoint to models/cropvision_cv.pt.
7. Save class names to models/class_names.json.
8. Print train loss, validation loss, validation accuracy, and macro F1 each epoch.
9. Include an evaluation script that creates:
   - confusion matrix
   - classification report
   - accuracy
   - macro F1
10. Save evaluation figures to reports/figures/.

Grad-CAM requirements:
1. Implement Grad-CAM for the chosen CNN model.
2. Create a function that takes an image path and model checkpoint, then returns:
   - predicted class
   - confidence
   - top 3 predictions
   - Grad-CAM heatmap overlay image
3. Save sample Grad-CAM outputs to reports/figures/.
4. Make Grad-CAM work inside the Streamlit app.

Weather-risk component:
1. Create weather_features.py that can fetch daily weather data from NASA POWER API using latitude, longitude, start date, and end date.
2. Features to calculate:
   - total rainfall over last 7 days
   - average humidity over last 7 days, if available
   - average temperature
   - max temperature
   - number of heat stress days over 30 C
   - number of wet days
3. Because we may not have labels for disease risk, create a simple synthetic training script in train_weather_model.py that trains a RandomForestClassifier on simulated labels based on rainfall, humidity, and heat stress.
4. Save the weather model to models/weather_risk_model.joblib.
5. Make clear in the README that this is a demo risk model and not agronomic advice.

Multimodal prediction:
1. Create multimodal_predict.py that combines:
   - image model disease prediction
   - image confidence
   - weather risk score
2. Final output should include:
   - predicted disease
   - image confidence
   - weather risk level: low, medium, high
   - combined risk summary
3. Keep the logic simple and transparent.

Streamlit app requirements:
1. App title: CropVision
2. User can upload a leaf image.
3. User can optionally enter:
   - latitude
   - longitude
   - start date
   - end date
4. App displays:
   - uploaded image
   - top prediction
   - top 3 predictions with confidence
   - Grad-CAM overlay
   - optional weather risk score
   - final combined interpretation
5. Add warning text:
   “This tool is for educational ML demonstration only, not professional crop diagnosis.”
6. Make the app robust if model files are missing. It should tell the user to train the model first instead of crashing.

README requirements:
Write a professional README with:
1. Project overview
2. Why this project matters for agtech and ML
3. Features
4. Dataset instructions
5. Setup instructions
6. Training command
7. Evaluation command
8. Weather model command
9. App command
10. Example portfolio bullets for resume
11. Limitations
12. Future improvements

Coding quality:
- Use argparse for scripts.
- Use pathlib instead of hardcoded string paths.
- Add helpful error messages.
- Keep functions modular.
- Include type hints where useful.
- Make it run even if optional weather model is not trained yet.
- Avoid huge dependencies unless necessary.
- Use matplotlib, not seaborn.
- Use joblib for saving the weather model.

After creating files:
1. Run basic import checks.
2. Run `python -m compileall src app`.
3. Fix any syntax/import errors.
4. Tell me the exact commands to run next.
