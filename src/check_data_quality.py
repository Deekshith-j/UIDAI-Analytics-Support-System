import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import load_all_data

def check_quality(data, report_path):
    with open(report_path, 'w', encoding='utf-8') as f:
        for key, df in data.items():
            f.write(f"\n{'='*20} {key.upper()} DATA QUALITY {'='*20}\n")
            
            # Missing values
            f.write("\n--- Missing Values ---\n")
            missing = df.isnull().sum()[df.isnull().sum() > 0]
            if not missing.empty:
                f.write(missing.to_string() + "\n")
            else:
                f.write("No missing values.\n")
            
            # Duplicates
            f.write(f"\n--- Duplicate Rows: {df.duplicated().sum()} ---\n")
            
            # Unique States
            if 'state' in df.columns:
                states = sorted(df['state'].astype(str).unique())
                f.write(f"\n--- Unique States ({len(states)}) ---\n")
                f.write(", ".join(states) + "\n")
                
            # Date range
            if 'date' in df.columns:
                f.write(f"\n--- Date Range: {df['date'].min()} to {df['date'].max()} ---\n")

if __name__ == "__main__":
    base_dir = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
    report_file = os.path.join(base_dir, "reports", "data_quality.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    print("Loading data...")
    data = load_all_data(base_dir)
    print("Generating report...")
    check_quality(data, report_file)
    print(f"Report saved to {report_file}")
