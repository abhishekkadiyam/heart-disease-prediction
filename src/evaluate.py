import joblib
from src.data_utils import load_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model_path, scaler_path, data_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = load_data(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))
    print("Confusion matrix:\n", confusion_matrix(y, preds))
