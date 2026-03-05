# predict.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def predict_crop(temperature, humidity, ph):
    """
    Predict crop recommendation based on climate parameters
    Returns top 5 predictions with confidence percentages
    """
    
    # Get base directory and load models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load model and encoder
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Create feature array (only temperature, humidity, pH)
    features = np.array([[temperature, humidity, ph]])
    
    # Get probability predictions for all classes
    probabilities = rf_model.predict_proba(features)[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_crops = le.inverse_transform(top_5_indices)
    top_5_confidences = probabilities[top_5_indices] * 100
    
    # Return as list of tuples (crop, confidence)
    top_5_predictions = [(crop, conf) for crop, conf in zip(top_5_crops, top_5_confidences)]
    
    return top_5_predictions

def predict_batch(data_path):
    """
    Predict crops for multiple samples from CSV
    Creates comparison dataframe like in notebook
    """
    
    # Get base directory and load models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load model and encoder
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Features (only temperature, humidity, pH)
    x = df[['temperature', 'humidity', 'ph']]
    
    # Make predictions 
    y_pred = rf_model.predict(x)
    
    # Create comparison dataframe 
    comparison_df = pd.DataFrame({
        'Index': range(len(x)),
        'Temperature': x['temperature'].values,
        'Humidity': x['humidity'].values,
        'pH': x['ph'].values,
        'Predicted (encoded)': y_pred,
        'Predicted (crop)': le.inverse_transform(y_pred)
    })
    
    # If actual labels exist in the data, add comparison 
    if 'label' in df.columns:
        le_label = le.transform(df['label'])
        comparison_df.insert(1, 'Actual (encoded)', le_label)
        comparison_df.insert(2, 'Actual (crop)', df['label'].values)
        comparison_df['Correct'] = le_label == y_pred
        
        # Print summary (same as notebook)
        print("\n" + "=" * 70)
        correct_count = comparison_df['Correct'].sum()
        incorrect_count = len(comparison_df) - correct_count
        print(f"SUMMARY: {correct_count} correct, {incorrect_count} incorrect out of {len(comparison_df)} total samples")
    
    return comparison_df

if __name__ == "__main__":
    print("=" * 70)
    print("🌱 CROP RECOMMENDATION - PREDICTION")
    print("=" * 70)
    
    print("\n📝 EXAMPLE: Prediction with Actual Test Data")
    print("-" * 70)
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'Crop_recommendation.csv')
    
    # Load the original data (like in notebook)
    df = pd.read_csv(data_path)
    
    # Load encoder
    models_dir = os.path.join(base_dir, 'models')
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
  
    np.random.seed(None)  # None = different random sample each run
                          # change to a specific numbmber if want to test
    sample_idx = np.random.randint(0, len(df))
    
    print(f"\n🔍 Using sample index: {sample_idx} (out of {len(df)} total samples)")
    
    temperature = df.loc[sample_idx, 'temperature']
    humidity = df.loc[sample_idx, 'humidity']
    ph = df.loc[sample_idx, 'ph']
    actual_crop = df.loc[sample_idx, 'label']  # Actual value from dataset (like y_test)
    
    # Make prediction
    top_5_predictions = predict_crop(temperature, humidity, ph)
    
    print(f"\nInput Parameters:")
    print(f"  Temperature={temperature}°C, Humidity={humidity}%, pH={ph}")
    print(f"\n✅ TOP 5 PREDICTIONS:")
    for i, (crop, confidence) in enumerate(top_5_predictions, 1):
        print(f"   {i}. {crop}: {confidence:.2f}%")
    print(f"\n   Actual value: {actual_crop}")
    print(f"   Match: {'✓ Correct' if top_5_predictions[0][0] == actual_crop else '✗ Incorrect'}")
    
    print("\n" + "=" * 70)
    print("💡 To predict from a CSV file, use:")
    print("   results = predict_batch('path/to/your/data.csv')")
    print("=" * 70)
