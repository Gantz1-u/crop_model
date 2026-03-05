# Plant Model - Crop Recommendation System

A machine learning system that recommends optimal crops based on climate parameters using Random Forest classification.

## Model Performance

- **Accuracy**: ~70% on test data
- **Algorithm**: Random Forest Classifier
- **Features**: Temperature, Humidity, pH (3 features)
- **Crops Supported**: 22 different crops
- **Output**: Top 5 crop recommendations with confidence percentages

## Project Structure

```
Plant_model/
│
├── data/
│   └── Crop_recommendation.csv           # Original dataset (2200 samples)
│
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── train_model.py                      # Model training script
│   ├── evaluate.py                          # Model evaluation script
│   └── predict.py                           # Prediction script
│
├── models/
│   ├── random_forest_model.pkl              # Trained model
│   └── label_encoder.pkl                     # Label encoder
│
├── templates/
│   ├── index.html                            # Web form interface
│   └── result.html                            # Prediction results page
│
├── notebooks/
│   ├── plant_model_v1.ipynb                  # Original development notebook
│   └── plant_model_v2.ipynb                  # Notebook v2
│
├── reports/
│   └── (Generated visualizations)             # Confusion matrix, feature importance
│
├── tests/
│   └── test_data.csv                          # Sample data for testing
│
├── app.py                                      # Flask web API server
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 1. Train the Model

```bash
python src/train_model.py
```

This will:
- Load data from `data/Crop_recommendation.csv`
- Train a Random Forest model
- Save the model and encoder to `models/`

### 2. Evaluate the Model

```bash
python src/evaluate.py
```

This will:
- Load the trained model
- Regenerate test data (20% split, same as training)
- Display accuracy, classification report, and feature importance
- Show sample predictions and misclassified samples

### 3. Make Predictions

**Command Line:**
```bash
python src/predict.py
```

This will:
- Load the trained model
- Make a prediction on a random sample from the dataset
- Show predicted vs actual crop

**Web Interface:**
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:5000`
- Fill in climate parameters (temperature, humidity, pH) in the web form
- Get instant top 5 crop recommendations with confidence percentages
- User-friendly interface for non-technical users

#### Use in Your Code

```python
from src.predict import predict_crop, predict_batch

# Single prediction - returns top 5 predictions with confidence
top_5_predictions = predict_crop(
    temperature=25.5, 
    humidity=80.0, 
    ph=6.5
)
for crop, confidence in top_5_predictions:
    print(f"{crop}: {confidence:.2f}%")

# Batch prediction from CSV
results = predict_batch('tests/test_data.csv')
print(results)
```

## Web API

The project includes a Flask web application for easy access to predictions.

### Start the Server

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

### Web Interface

Access the web form at `http://127.0.0.1:5000` to:
- Enter climate data (temperature, humidity, pH)
- Get top 5 crop recommendations with confidence percentages instantly

### For Developers


**Example API Request:**
```python
import requests

data = {
    "temperature": 25.5,
    "humidity": 70,
    "ph": 6.5
}

response = requests.post('http://127.0.0.1:5000/api/predict', json=data)
print(response.json())  # Returns top 5 predictions with confidence percentages
```
flask >= 3.0.0 (for web API)
- flask-cors >= 6.0.0 (for API CORS support)
- 

## Features

**Input Parameters:**
- **temperature** - Temperature in Celsius
- **humidity** - Relative humidity in percentage
- **ph** - pH value of soil

**Output:**
- Top 5 recommended crops with confidence percentages from 22 options (rice, maize, banana, etc.)

## Model Details

**Algorithm**: Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

**Feature Importance:**
1. Temperature
2. Humidity
3. pH

*Note: Using only 3 climate features (temperature, humidity, pH) results in lower accuracy (~70%) compared to using soil nutrient features, but provides a simpler model that focuses on climate conditions.*

## Supported Crops

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Evaluation Results

- **Overall Accuracy**: ~70% on test data
- **Prediction Method**: Top 5 recommendations with confidence percentages
- **Model Trade-off**: Simplified feature set (temperature, humidity, pH only) for easier data collection and deployment

