# evaluate.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load the trained model and regenerate test data"""
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    
    # Paths
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    data_path = os.path.join(data_dir, 'Crop_recommendation.csv')
    
    print("=" * 70)
    print("📊 LOADING MODEL FOR EVALUATION")
    print("=" * 70)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please run train_model.py first!")
        return None
    
    # Load model and encoder
    rf_model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    # Regenerate test data (same split as training)
    df = pd.read_csv(data_path)
    df["le_label"] = le.transform(df['label'])  # Use existing encoder
    
    x = df[['K', 'P', 'N', 'temperature', 'humidity', 'ph']]
    y = df['le_label']
    
    # Same split as in train_model.py (test_size=0.2, random_state=42, stratify=y)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=.2, random_state=42, stratify=y
    )
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Encoder loaded from: {encoder_path}")
    print(f"✅ Test data regenerated: {len(x_test)} samples")
    
    return rf_model, le, x_test, y_test

def evaluate_model(rf_model, le, x_test, y_test):
    """Evaluate model exactly like your original code"""
    
    print("\n" + "=" * 70)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    # Make predictions
    y_pred = rf_model.predict(x_test)
    
    # ACCURACY - Same as your original
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Random Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # CLASSIFICATION REPORT - Same as your original
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # FEATURE IMPORTANCE - Same as your original
    feature_importance = pd.DataFrame({
        'feature': x_test.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    print("\n" + "=" * 70)
    
    return y_pred

def show_sample_predictions(y_test, y_pred, le):
    """Show sample predictions exactly like your original code"""
    
    # Create comparison dataframe - EXACTLY like your original
    comparison_df = pd.DataFrame({
        'Index': range(len(y_test)),
        'Actual (encoded)': y_test,
        'Actual (crop)': le.inverse_transform(y_test),
        'Predicted (encoded)': y_pred,
        'Predicted (crop)': le.inverse_transform(y_pred),
        'Correct': y_test == y_pred
    })
    
    # Show first 20 samples - exactly like your original
    print("\nSample Predictions (First 20):")
    print(comparison_df.head(20).to_string(index=False))
    
    # Summary of correct/incorrect - exactly like your original
    print("\n" + "=" * 70)
    correct_count = comparison_df['Correct'].sum()
    incorrect_count = len(comparison_df) - correct_count
    print(f"SUMMARY: {correct_count} correct, {incorrect_count} incorrect out of {len(comparison_df)} total samples")
    
    return comparison_df

def show_prediction_summary(comparison_df):
    """Show detailed summary of predictions"""
    
    print("\n" + "=" * 70)
    print("📈 PREDICTION SUMMARY")
    print("=" * 70)
    
    # Accuracy by crop type
    print("\nAccuracy by Crop Type:")
    print("-" * 50)
    
    # Group by actual crop and calculate accuracy
    crop_accuracy = comparison_df.groupby('Actual (crop)')['Correct'].agg(['count', 'sum'])
    crop_accuracy['accuracy'] = (crop_accuracy['sum'] / crop_accuracy['count'] * 100).round(2)
    crop_accuracy = crop_accuracy.sort_values('accuracy', ascending=False)
    
    for crop, row in crop_accuracy.iterrows():
        print(f"  {crop:15s}: {row['accuracy']:.1f}% ({int(row['sum'])}/{int(row['count'])})")
    
    # Find misclassified samples
    misclassified = comparison_df[~comparison_df['Correct']]
    if len(misclassified) > 0:
        print("\n❌ Misclassified Samples (First 10):")
        print("-" * 50)
        for idx, row in misclassified.head(10).iterrows():
            print(f"  Sample {int(row['Index'])}: Actual={row['Actual (crop)']} → Predicted={row['Predicted (crop)']}")

def main():
    """Main evaluation function"""
    
    # Load model and data
    result = load_model_and_data()
    if result is None:
        return
    
    rf_model, le, x_test, y_test = result
    
    # Evaluate model (exactly like your original)
    y_pred = evaluate_model(rf_model, le, x_test, y_test)
    
    # Show sample predictions (exactly like your original)
    comparison_df = show_sample_predictions(y_test, y_pred, le)
    
    # Show additional summary (extra but useful)
    show_prediction_summary(comparison_df)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()