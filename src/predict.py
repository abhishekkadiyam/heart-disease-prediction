import joblib
import numpy as np

def predict_from_list(model_path, scaler_path, features_list):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X = np.array(features_list).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return int(pred[0])
