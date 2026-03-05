# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def train_and_save_model(data_path='Crop_recommendation.csv'):
    """Train model and save to disk"""
    
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', data_path)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Encode labels
    le = LabelEncoder()
    df["le_label"] = le.fit_transform(df['label'])
    
    # Features and target (only temperature, humidity, pH)
    x = df[['temperature', 'humidity', 'ph']]
    y = df['le_label']
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=42, stratify=y
    )
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(x_train, y_train)
    
    # Save model and encoder
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))
    
    print("✅ Model and encoder saved successfully!")
    return rf_model, le, x_test, y_test

if __name__ == "__main__":
    train_and_save_model(data_path='Crop_recommendation.csv') 