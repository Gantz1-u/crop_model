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
    # Get form data (only temperature, humidity, pH)
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
    features = np.array([[temperature, humidity, ph]])
    
    # Get probability predictions for all classes
    probabilities = rf_model.predict_proba(features)[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_crops = le.inverse_transform(top_5_indices)
    top_5_confidences = probabilities[top_5_indices] * 100
    
    # Create list of tuples (crop, confidence)
    top_5_predictions = [(crop, round(conf, 2)) for crop, conf in zip(top_5_crops, top_5_confidences)]
    
    # Render result template with top 5 predictions
    return render_template('result.html', predictions=top_5_predictions,
                         temperature=temperature, humidity=humidity, ph=ph)

if __name__ == '__main__':
    app.run(debug=True)   