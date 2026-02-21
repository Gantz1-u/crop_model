import numpy as np
from flask import Flask, request, render_template
import joblib
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    K = float(request.form['K'])
    P = float(request.form['P'])
    N = float(request.form['N'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    
    # Load model and encoder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    
    rf_model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    # Feature array and make prediction
    features = np.array([[K, P, N, temperature, humidity, ph]])
    pred_encoded = rf_model.predict(features)[0]
    pred_crop = le.inverse_transform([pred_encoded])[0]
    
    # Render result template
    return render_template('result.html', crop=pred_crop, K=K, P=P, N=N, 
                         temperature=temperature, humidity=humidity, ph=ph)

if __name__ == '__main__':
    app.run(debug=True)   