import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from features import create_enrolment_features, create_demographic_features, create_biometric_features, create_composite_features, calculate_performance_score

BASE_DIR = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def run_feature_pipeline():
    print("Loading Cleaned Data...")
    try:
        enrol_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'enrolment_cleaned.parquet'))
        demo_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'demographic_cleaned.parquet'))
        bio_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'biometric_cleaned.parquet'))
    except FileNotFoundError as e:
        print(f"Error loading parquet files: {e}")
        return

    print("Generating Enrolment Features...")
    enrol_feat = create_enrolment_features(enrol_df)
    
    print("Generating Demographic Features...")
    demo_feat = create_demographic_features(demo_df)
    
    print("Generating Biometric Features...")
    bio_feat = create_biometric_features(bio_df)
    
    print("Generating Composite Features...")
    composite_feat = create_composite_features(enrol_feat, demo_feat, bio_feat)
    
    print("Calculating Performance Score...")
    composite_feat['performance_score'] = calculate_performance_score(composite_feat)
    
    print("Saving Features...")
    enrol_feat.to_parquet(os.path.join(PROCESSED_DIR, 'enrolment_features.parquet'), index=False)
    demo_feat.to_parquet(os.path.join(PROCESSED_DIR, 'demographic_features.parquet'), index=False)
    bio_feat.to_parquet(os.path.join(PROCESSED_DIR, 'biometric_features.parquet'), index=False)
    composite_feat.to_parquet(os.path.join(PROCESSED_DIR, 'composite_features.parquet'), index=False)
    
    print("Feature Engineering Complete.")
    print(f"Composite Data Shape: {composite_feat.shape}")
    print(f"Sample Score: {composite_feat['performance_score'].head()}")

if __name__ == "__main__":
    run_feature_pipeline()
