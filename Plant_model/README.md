
## Model Performance

- **Accuracy**: ~76% on test data
- **Algorithm**: Random Forest Classifier
- **Features**: temperature, humidity, pH (reduced from original 6 features)
- **Crops Supported**: 22 different crops
- **Note**: Accuracy reduced from 97% after removing K, P, N features for simplicity

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

**Key Dependencies:**
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning (Random Forest)
- `flask` - Web server and REST API
- `flask-cors` - Enable CORS for JavaScript fetch requests
- `joblib` - Model serialization

**Requirements File (`requirements.txt`):**
```
pandas>=2.0.0
numpy>=1.23.0
scikit-learn>=1.3.0
joblib>=1.3.0
flask>=3.0.0
flask-cors>=6.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
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
- Display top 15 crop recommendations with probabilities

**Web Interface:**
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:5000`
- Modern interface with HTML, CSS, and JavaScript
- Uses AJAX to communicate with JSON API
- Real-time predictions without page reload
- Displays top 15 crop recommendations with probabilities

#### Use in Your Code

```python
from src.predict import predict_crop, predict_top_n_crops, predict_batch

# Single prediction (returns top recommendation)
encoded, crop = predict_crop(
    temperature=25.5, 
    humidity=80.0, 
    ph=6.5
)
print(f"Recommended crop: {crop}")

# Get top 15 crop recommendations with probabilities (default)
top_crops = predict_top_n_crops(
    temperature=25.5, 
    humidity=80.0, 
    ph=6.5
)
print(top_crops)
# Output:
#          crop  probability
# 0        rice        0.85
# 1       maize        0.08
# 2      cotton        0.03
# ...       ...         ...
# 14     papaya        0.001

# Get specific number (e.g., top 5)
top_5 = predict_top_n_crops(temperature=25.5, humidity=80.0, ph=6.5, top_n=5)

# Get all 22 crops
all_crops = predict_top_n_crops(temperature=25.5, humidity=80.0, ph=6.5, top_n=22)

# Batch prediction from CSV
results = predict_batch('tests/test_data.csv')
print(results)
```

## Web API

The project includes a Flask web application with a JSON REST API for JavaScript communication.

### Architecture

```
Browser (JavaScript)  ←→  Flask Server (Python)  ←→  ML Model (scikit-learn)
     Port: Any              Port: 5000               Random Forest
     
User fills form
    ↓
JavaScript fetch('/api/predict')
    ↓
Flask processes request
    ↓
Model predicts top 15 crops
    ↓
JSON response with probabilities
    ↓
JavaScript displays results
```

### Start the Server

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

**Important:** Always access the web app at **port 5000** (Flask), not 5500 (Live Server)

## Features

**Input Parameters:**
- **temperature** - Temperature in Celsius
- **humidity** - Relative humidity in percentage
- **ph** - pH value of soil

**Output Options:**

1. **Single Crop Prediction** (`predict_crop`)
   - Returns: The top recommended crop
   - Use case: When you need one best recommendation

2. **Top N Crop Recommendations** (`predict_top_n_crops`)
   - Returns: DataFrame with top N crops and their probabilities
   - Default: Top 15 recommendations (customizable to any number)
   - Probabilities: Shows confidence level for each recommendation
   - Use case: When you want multiple options ranked by suitability
   - Stored in variable for easy access by other developers
   - Example output:
     ```
     Rank  Crop          Probability
     1.    rice          85.2%
     2.    maize         8.3%
     3.    cotton        3.1%
     ...
     15.   papaya        0.1%
     ```

3. **Batch Predictions** (`predict_batch`)
   - Input: CSV file with multiple samples
   - Returns: DataFrame with predictions for all samples
   - Use case: Process multiple predictions at once

## Model Details

**Algorithm**: Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

**Feature Importance:**
1. Humidity: ~40%
2. Temperature: ~35%
3. pH: ~25%


## Understanding Model Accuracy vs Prediction Confidence

### Model Accuracy (76%)
- **Measures**: How often the model predicts correctly across all test samples
- **Calculation**: `(Correct predictions / Total predictions) × 100`
- **Means**: Out of 100 predictions, about 76 will have the correct crop as the top recommendation

### Prediction Confidence/Probability
- **Measures**: How confident the model is about a specific prediction
- **Range**: 0% to 100% for each crop
- **Example**: "Kidneybeans - 58.38%" means 58.38% confidence for this specific input


### How to Access the Web App

**✅ CORRECT Way:**
1. Start Flask server: `python app.py`
2. Open browser to: **http://127.0.0.1:5000/** (port 5000)
3. Fill in the form and get instant recommendations

**❌ COMMON MISTAKE:**
- **Do NOT use** Live Server (port 5500) or open HTML file directly
- **Do NOT access** `127.0.0.1:5500/templates/index.html`
- The JSON API endpoint (`/api/predict`) only exists on the Flask server (port 5000)

### Troubleshooting

#### Error: "405 Method Not Allowed"
**Cause**: You're accessing the wrong port (likely 5500 instead of 5000)
**Solution**: 
1. Check your browser URL - should be `http://127.0.0.1:5000/`
2. Close any Live Server instances in VS Code
3. Access only through Flask server on port 5000

#### Error: "Failed to fetch" or "Connection refused"
**Cause**: Flask server is not running
**Solution**:
1. Run `python app.py` in terminal
2. Wait for message: "Running on http://127.0.0.1:5000"
3. Then access in browser

#### Error: "CORS policy blocked"
**Cause**: Missing or incorrect CORS configuration
**Solution**: 
1. Verify `flask-cors` is installed: `pip install flask-cors`
2. Check `app.py` imports: `from flask_cors import CORS`
3. Restart Flask server

#### Browser shows old version after updating code
**Solution**:
1. Hard refresh: Press **Ctrl + Shift + R** (or Ctrl + F5)
2. Or clear browser cache
3. Or use incognito/private mode

## Supported Crops

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Evaluation Results


### Why 76% Accuracy?

The model originally used 6 features (N, P, K, temperature, humidity, pH) achieving 97% accuracy. After removing the 3 soil nutrient features:


### For Development
```bash
python app.py  # Runs on http://127.0.0.1:5000 with debug mode
```


