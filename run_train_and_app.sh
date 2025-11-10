#!/usr/bin/env bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Add your data/heart.csv and then run:"
echo "python src/train.py --data data/heart.csv --out models/model.pkl --scaler models/scaler.pkl"
echo "Then run the app: python app/app.py"
