# predict.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def predict_crop(K, P, N, temperature, humidity, ph):
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
    
    # Create feature array (same order as training: K, P, N, temperature, humidity, ph)
    features = np.array([[K, P, N, temperature, humidity, ph]])
    
    # Make prediction 
    pred_encoded = rf_model.predict(features)[0]
    pred_crop = le.inverse_transform([pred_encoded])[0]
    
    return pred_encoded, pred_crop

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
    x = df[['K', 'P', 'N', 'temperature', 'humidity', 'ph']]
    
    # Make predictions 
    y_pred = rf_model.predict(x)
    
    # Create comparison dataframe 
    comparison_df = pd.DataFrame({
        'Index': range(len(x)),
        'K': x['K'].values,
        'P': x['P'].values,
        'N': x['N'].values,
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
    
    K = df.loc[sample_idx, 'K']
    P = df.loc[sample_idx, 'P']
    N = df.loc[sample_idx, 'N']
    temperature = df.loc[sample_idx, 'temperature']
    humidity = df.loc[sample_idx, 'humidity']
    ph = df.loc[sample_idx, 'ph']
    actual_crop = df.loc[sample_idx, 'label']  # Actual value from dataset (like y_test)
    
    # Make prediction
    encoded, predicted_crop = predict_crop(K, P, N, temperature, humidity, ph)
    
    print(f"\nInput Parameters:")
    print(f"  K={K}, P={P}, N={N}")
    print(f"  Temperature={temperature}°C, Humidity={humidity}%, pH={ph}")
    print(f"\n✅ PREDICTION RESULT:")
    print(f"   Predicted value: {predicted_crop}")
    print(f"   Actual value:    {actual_crop}")
    print(f"   Match: {'✓ Correct' if predicted_crop == actual_crop else '✗ Incorrect'}")
    
    print("\n" + "=" * 70)
    print("💡 To predict from a CSV file, use:")
    print("   results = predict_batch('path/to/your/data.csv')")
    print("=" * 70)
