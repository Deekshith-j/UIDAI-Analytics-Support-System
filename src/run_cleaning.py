import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import load_all_data
from cleaning import clean_dataset

BASE_DIR = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def generate_data_dictionary(df, name):
    print(f"\n--- Data Dictionary for {name} ---")
    print(f"{'Column':<30} {'Type':<15} {'Description'}")
    print("-" * 60)
    for col in df.columns:
        dtype = str(df[col].dtype)
        desc = "Unique Identifier" if "id" in col else "Metric/Attribute"
        if "date" in col: desc = "Temporal"
        if "state" in col or "district" in col: desc = "Geographic"
        print(f"{col:<30} {dtype:<15} {desc}")

def run_pipeline():
    print("Loading Raw Data...")
    data = load_all_data(BASE_DIR)
    
    cleaned_data = {}
    
    for key, df in data.items():
        print(f"\nProcessing {key}...")
        clean_df = clean_dataset(df)
        cleaned_data[key] = clean_df
        
        # Save
        output_path = os.path.join(PROCESSED_DIR, f"{key}_cleaned.parquet")
        clean_df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")
        
        # Generate Dictionary
        generate_data_dictionary(clean_df, key)
        
if __name__ == "__main__":
    run_pipeline()
