# 🌱 Plant Model - Crop Recommendation System

A machine learning system that recommends optimal crops based on soil and climate parameters using Random Forest classification.

## 📊 Model Performance

- **Accuracy**: 97.05% on test data
- **Algorithm**: Random Forest Classifier
- **Features**: K, P, N, temperature, humidity, pH
- **Crops Supported**: 22 different crops

## 📁 Project Structure

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

```bash
python src/predict.py
```

This will:
- Load the trained model
- Make a prediction on a random sample from the dataset
- Show predicted vs actual crop

#### Use in Your Code

```python
from src.predict import predict_crop, predict_batch

# Single prediction
encoded, crop = predict_crop(
    K=20, P=40, N=60, 
    temperature=25.5, 
    humidity=80.0, 
    ph=6.5
)
print(f"Recommended crop: {crop}")

# Batch prediction from CSV
results = predict_batch('tests/test_data.csv')
print(results)
```

## 📈 Features

**Input Parameters:**
- **N** - Nitrogen content in soil
- **P** - Phosphorus content in soil
- **K** - Potassium content in soil
- **temperature** - Temperature in Celsius
- **humidity** - Relative humidity in percentage
- **ph** - pH value of soil

**Output:**
- Recommended crop from 22 options (rice, maize, banana, etc.)

## 🎯 Model Details

**Algorithm**: Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

**Feature Importance (Top 3):**
1. Humidity: 26.5%
2. Potassium (K): 22.1%
3. Phosphorus (P): 17.4%

## 📊 Supported Crops

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## 🔬 Evaluation Results

- **Overall Accuracy**: 97.05%
- **Perfect Predictions (100%)**: 16 out of 22 crops
- **Lowest Accuracy**: Lentil at 75% (some confusion with blackgram and pigeonpeas)

## 📝 Development

The project was initially developed in Jupyter notebooks (`notebooks/plant_model_v1.ipynb`) and then refactored into modular Python scripts for better organization and reusability.

## 🛠️ Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.23.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- matplotlib >= 3.7.0 (for visualizations)
- seaborn >= 0.12.0 (for visualizations)


