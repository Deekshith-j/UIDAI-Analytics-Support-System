import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models import prepare_time_series_data, train_forecast_model, train_risk_model

BASE_DIR = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

def run_models():
    print("Loading Composite Data...")
    try:
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'composite_features.parquet'))
    except FileNotFoundError:
        print("Data not found.")
        return

    # 1. Forecasting Workload (Total Activity)
    print("\n--- Forecasting Workload ---")
    ts_data = prepare_time_series_data(df, 'total_activity')
    if not ts_data.empty:
        model, rmse, preds, y_test = train_forecast_model(ts_data, 'total_activity')
        print(f"Workload Forecast RMSE: {rmse:.2f}")
        
        # Save forecast chart data or results
        results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
        results.to_csv(os.path.join(REPORT_DIR, 'forecast_results.csv'))
        print(f"Forecast results saved to {REPORT_DIR}")
    else:
        print("Not enough time series data for forecasting.")

    # 2. Risk Classification
    print("\n--- Risk Classification ---")
    # Need to ensure columns exist. 'child_to_adult_bio_ratio' might be missing if we didn't merge it well.
    # composite_features.parquet was created from enrolment, demo, bio. Bio has the ratio.
    # Let's check columns in `features.py`. Yes, `child_to_adult_bio_ratio` is in bio_feat.
    # But composite_feat is merged. We should check if it's there.
    # In `features.py`, I did: m = merge(e, d); m = merge(m, b).
    # `b` (total_bio_updates) was selected. I didn't select `child_to_adult_bio_ratio` in the merge subset!
    # I only selected: b = bio_df[['date', 'state', 'district', 'pincode', 'total_bio_updates']].copy()
    # Solution: I should rely on 'total_enrolment' and 'enrolment_to_update_ratio' which ARE in composite.
    
    try:
        clf, report, feats = train_risk_model(df)
        print(f"Model Accuracy Report:\n{report}")
        print(f"Features used: {feats}")
    except Exception as e:
        print(f"Error in risk modeling: {e}")

if __name__ == "__main__":
    run_models()
