# Heart Disease Prediction (AI/ML)

Predict whether a patient has heart disease using classical ML models and a simple Flask demo.

**Author / GitHub:** abhishekkadiyam

## Summary
This repository contains code to preprocess data, train a Random Forest model, and run a Flask web demo that accepts clinical inputs and returns a prediction.

**Reported model accuracy (example): 87%**

## Repo structure
- `data/` - **(empty)** place `heart.csv` here (see `data/README.md`)
- `src/` - preprocessing, training, and helper scripts
- `app/` - Flask web app for demo
- `models/` - saved models (gitignored)
- `notebooks/` - optional Jupyter notebooks (not included)

## Quickstart (local)
1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your dataset:
- Place your `heart.csv` inside the `data/` folder. The script expects a column named `target` for labels.

4. Train the model:
```bash
python src/train.py --data data/heart.csv --out models/model.pkl --scaler models/scaler.pkl
```

5. Run the web demo:
```bash
python app/app.py
# open http://127.0.0.1:5000 in your browser
```


