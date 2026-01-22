import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from anomaly_detection import analyze_anomalies

BASE_DIR = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

def run_anomaly_pipeline():
    print("Loading Composite Data...")
    try:
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'composite_features.parquet'))
    except FileNotFoundError:
        print("Composite data not found.")
        return

    print("Running Anomaly Detection...")
    stat_anoms, ml_anoms = analyze_anomalies(df)
    
    # Save Reports
    stat_anoms.to_csv(os.path.join(REPORT_DIR, 'statistical_anomalies.csv'), index=False)
    ml_anoms.to_csv(os.path.join(REPORT_DIR, 'ml_anomalies.csv'), index=False)
    print(f"Anomaly reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    run_anomaly_pipeline()
