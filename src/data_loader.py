import pandas as pd
import glob
import os
from pathlib import Path

def load_dataset(directory_path, file_pattern="*.csv"):
    """
    Loads all CSV files from a directory matching the pattern and concatenates them.
    Adds a 'source_file' column to track origin.
    """
    all_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not all_files:
        print(f"No files found in {directory_path} matching {file_pattern}")
        return pd.DataFrame()
    
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df['source_file'] = os.path.basename(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df

def normalize_columns(df):
    """
    Standardizes column names to snake_case and renames specific inconsistent columns.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Rename specific columns for consistency
    rename_map = {
        'bio_age_17_': 'bio_age_18_plus',
        'demo_age_17_': 'demo_age_18_plus',
        'age_18_greater': 'enrolment_age_18_plus',
        'age_0_5': 'enrolment_age_0_5',
        'age_5_17': 'enrolment_age_5_17',
        'bio_age_5_17': 'bio_age_5_17',
        'demo_age_5_17': 'demo_age_5_17'
    }
    
    df = df.rename(columns=rename_map)
    return df

def process_date_column(df, date_col='date'):
    """
    Converts date column to datetime objects.
    """
    if date_col in df.columns:
        # The format seen in the files is DD-MM-YYYY (e.g., 01-03-2025)
        df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
    return df

def load_all_data(base_path):
    """
    Loads all three datasets: Enrolment, Biometric, Demographic.
    Returns a dictionary of DataFrames.
    """
    data = {}
    
    datasets = {
        'enrolment': 'api_data_aadhar_enrolment',
        'biometric': 'api_data_aadhar_biometric',
        'demographic': 'api_data_aadhar_demographic'
    }
    
    for key, folder_name in datasets.items():
        print(f"Loading {key} data...")
        folder_path = os.path.join(base_path, folder_name)
        df = load_dataset(folder_path)
        df = normalize_columns(df)
        df = process_date_column(df)
        data[key] = df
        print(f"Loaded {key} data: {df.shape}")
        
    return data

if __name__ == "__main__":
    # Example usage
    base_dir = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
    data = load_all_data(base_dir)
