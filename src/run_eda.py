import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from eda_utils import set_style, plot_time_series, plot_state_comparison, plot_age_distribution

BASE_DIR = r"C:\Users\Deekshith J\OneDrive\Desktop\UDAI HACKATHON"
Processed_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(REPORT_DIR, exist_ok=True)

def run_eda():
    set_style()
    
    print("Loading datasets for EDA...")
    try:
        enrol_df = pd.read_parquet(os.path.join(Processed_DIR, 'enrolment_features.parquet'))
        demo_df = pd.read_parquet(os.path.join(Processed_DIR, 'demographic_features.parquet'))
        bio_df = pd.read_parquet(os.path.join(Processed_DIR, 'biometric_features.parquet'))
        comp_df = pd.read_parquet(os.path.join(Processed_DIR, 'composite_features.parquet'))
    except FileNotFoundError:
        print("Feature files not found. Run run_features.py first.")
        return

    print("Generating Plots...")
    
    # 1. Enrolment Trends
    plot_time_series(enrol_df, 'date', 'total_enrolment', 'Monthly Enrolment Trend', 
                     os.path.join(REPORT_DIR, 'enrolment_trend.png'))
    
    # 2. State-wise Enrolment
    plot_state_comparison(enrol_df, 'total_enrolment', 'Top 10 States by Enrolment', 
                          os.path.join(REPORT_DIR, 'state_enrolment.png'))
    
    # 3. Enrolment Age Distribution
    plot_age_distribution(enrol_df, ['enrolment_age_0_5', 'enrolment_age_5_17', 'enrolment_age_18_plus'], 
                          'Enrolment Age Distribution', os.path.join(REPORT_DIR, 'enrolment_age_dist.png'))
                          
    # 4. Demographic Updates vs Biometric Updates Trend
    # Need to aggregate first
    demo_ts = demo_df.set_index('date').resample('M')['total_demo_updates'].sum()
    bio_ts = bio_df.set_index('date').resample('M')['total_bio_updates'].sum()
    
    plt.figure()
    plt.plot(demo_ts.index, demo_ts.values, label='Demographic Updates', marker='o')
    plt.plot(bio_ts.index, bio_ts.values, label='Biometric Updates', marker='x')
    plt.title('Updates Trend Comparison')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'updates_comparison.png'))
    plt.close()
    
    # 5. Region Stress Index Top Districts
    # Group by district and sum stress index (or mean?) 
    # Stress index was calculated per row (pincode/day). Summing it gives total load.
    # We normalized data for performance score, but 'total_activity' is raw load.
    district_stress = comp_df.groupby(['state', 'district'])['total_activity'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(14, 8))
    # Create labels like "District, State"
    labels = [f"{idx[1]}, {idx[0]}" for idx in district_stress.index]
    import seaborn as sns
    sns.barplot(x=district_stress.values, y=labels, palette='magma')
    plt.title('Top 10 Districts by Total Activity Load (Stress)', fontsize=16)
    plt.xlabel('Total Transactions')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'district_stress.png'))
    plt.close()

    print(f"EDA plots saved to {REPORT_DIR}")

if __name__ == "__main__":
    run_eda()
