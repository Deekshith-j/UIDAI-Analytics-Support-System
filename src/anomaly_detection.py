import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_statistical_anomalies(df, col, threshold=3):
    """
    Detects anomalies using Z-score.
    """
    mean = df[col].mean()
    std = df[col].std()
    z_scores = (df[col] - mean) / std
    return df[np.abs(z_scores) > threshold].copy()

def detect_isolation_forest_anomalies(df, cols, contamination=0.01):
    """
    Detects anomalies using Isolation Forest.
    """
    data = df[cols].fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(scaled_data)
    
    # -1 is anomaly
    anomalies = df[preds == -1].copy()
    anomalies['anomaly_score'] = iso.decision_function(scaled_data)[preds == -1]
    return anomalies

def analyze_anomalies(df):
    """
    Runs anomaly detection on the composite dataset.
    """
    print("--- Statistical Anomalies (High Enrolment Volume) ---")
    stat_anomalies = detect_statistical_anomalies(df, 'total_enrolment')
    print(f"Found {len(stat_anomalies)} anomalies using Z-score > 3")
    if not stat_anomalies.empty:
        print(stat_anomalies[['date', 'state', 'district', 'total_enrolment']].head(10))
        
    print("\n--- ML-Based Anomalies (Isolation Forest) ---")
    features = ['total_enrolment', 'total_demo_updates', 'total_bio_updates', 'enrolment_to_update_ratio']
    ml_anomalies = detect_isolation_forest_anomalies(df, features)
    print(f"Found {len(ml_anomalies)} anomalies using Isolation Forest")
    
    if not ml_anomalies.empty:
        print("Top 10 Anomalies by Score (Lowest Score = Most Anomalous):")
        print(ml_anomalies[['state', 'district', 'date'] + features + ['anomaly_score']].sort_values('anomaly_score').head(10))
        
    return stat_anomalies, ml_anomalies
