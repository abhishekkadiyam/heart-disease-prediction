from flask import Flask, render_template, request
import joblib
import os
from src.predict import predict_from_list

app = Flask(__name__, template_folder='templates')

MODEL_PATH = os.path.join('..', 'models', 'model.pkl')
SCALER_PATH = os.path.join('..', 'models', 'scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure order matches training features
        feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        features = []
        for name in feature_names:
            val = request.form.get(name)
            if val is None or val == '':
                return render_template('index.html', result=f'Missing input for {name}')
            features.append(float(val))
    except Exception as e:
        return render_template('index.html', result=f'Invalid input: {e}')
    # Check model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return render_template('index.html', result='Model or scaler not found. Please train model and place files in models/ folder.')
    pred = predict_from_list(MODEL_PATH, SCALER_PATH, features)
    return render_template('index.html', result=('Heart Disease Detected' if pred == 1 else 'No Heart Disease'))

if __name__ == '__main__':
    app.run(debug=True)
