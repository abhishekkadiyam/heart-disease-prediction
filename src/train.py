#!/usr/bin/env python
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_utils import load_data
from src.preprocess import preprocess
import os

def main(args):
    df = load_data(args.data)
    X, y, scaler = preprocess(df, save_scaler=args.scaler if args.scaler else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(model, args.out)
        print("Saved model to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data CSV (heart.csv)")
    parser.add_argument("--out", help="path to save trained model (models/model.pkl)")
    parser.add_argument("--scaler", help="path to save scaler (models/scaler.pkl)")
    args = parser.parse_args()
    main(args)
