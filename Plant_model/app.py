import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for JavaScript requests

# Load models once at startup
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')

rf_model = joblib.load(model_path)
le = joblib.load(encoder_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for JavaScript to call
    Returns top 15 crop recommendations with probabilities
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        top_n = int(data.get('top_n', 15))  # Default to 15
        
        # Make prediction
        features = np.array([[temperature, humidity, ph]])
        
        # Get probabilities for all crops
        probabilities = rf_model.predict_proba(features)[0]
        
        # Get top N indices
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        
        # Create results list
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'crop': le.inverse_transform([idx])[0],
                'probability': float(probabilities[idx])
            })
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'input': {
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)   