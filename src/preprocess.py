import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(df, target_col='target', save_scaler=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if save_scaler:
        joblib.dump(scaler, save_scaler)
    return X_scaled, y, scaler
