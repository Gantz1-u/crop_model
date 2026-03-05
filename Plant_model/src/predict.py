# predict.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def predict_crop(temperature, humidity, ph):
    """
    Predict crop recommendation based on soil and climate parameters
    Same prediction logic as plant_model_v1.ipynb
    """
    
    # Get base directory and load models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load model and encoder
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Create feature array (same order as training: temperature, humidity, ph)
    features = np.array([[temperature, humidity, ph]])
    
    # Make prediction 
    pred_encoded = rf_model.predict(features)[0]
    pred_crop = le.inverse_transform([pred_encoded])[0]
    
    return pred_encoded, pred_crop

def predict_top_n_crops(temperature, humidity, ph, top_n=15):
    """
    Predict top N crop recommendations with probabilities
    
    Parameters:
    -----------
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage
    ph : float
        pH value
    top_n : int, default=15
        Number of top crop recommendations to return
    
    Returns:
    --------
    DataFrame with columns: crop, probability (sorted by probability descending)
    """
    
    # Get base directory and load models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load model and encoder
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Create feature array
    features = np.array([[temperature, humidity, ph]])
    
    # Get probability predictions for all classes
    probabilities = rf_model.predict_proba(features)[0]
    
    # Get top N indices
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    
    # Create results dataframe
    results = pd.DataFrame({
        'crop': le.inverse_transform(top_indices),
        'probability': probabilities[top_indices]
    })
    
    return results

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
    
    # Features (same as notebook)
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
    encoded, predicted_crop = predict_crop(temperature, humidity, ph)
    
    print(f"\nInput Parameters:")
    print(f"  Temperature={temperature}°C, Humidity={humidity}%, pH={ph}")
    print(f"\n✅ PREDICTION RESULT:")
    print(f"   Predicted value: {predicted_crop}")
    print(f"   Actual value:    {actual_crop}")
    print(f"   Match: {'✓ Correct' if predicted_crop == actual_crop else '✗ Incorrect'}")
    
    # Show top 15 crop recommendations
    print("\n" + "=" * 70)
    print("📊 TOP 15 CROP RECOMMENDATIONS (with probabilities):")
    print("-" * 70)
    top_crops_result = predict_top_n_crops(temperature, humidity, ph, top_n=15)
    for idx, row in top_crops_result.iterrows():
        print(f"   {idx+1}. {row['crop']:15s} - {row['probability']*100:5.2f}%")
    
    print("\n" + "=" * 70)
    print("💡 Usage Examples:")
    print("   # Get top 15 crops with probabilities (default):")
    print("   top_crops = predict_top_n_crops(temp, humidity, ph)")
    print("\n   # Get specific number of recommendations:")
    print("   top_5 = predict_top_n_crops(temp, humidity, ph, top_n=5)")
    print("\n   # Batch predictions from CSV:")
    print("   results = predict_batch('path/to/your/data.csv')")
    print("=" * 70)
