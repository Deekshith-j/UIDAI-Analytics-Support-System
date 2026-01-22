import pandas as pd
import numpy as np

def create_enrolment_features(df):
    """
    Creates derived features for Enrolment data.
    """
    df = df.copy()
    
    # Total Enrolment
    df['total_enrolment'] = df['enrolment_age_0_5'] + df['enrolment_age_5_17'] + df['enrolment_age_18_plus']
    
    # Age Group Shares (Avoid division by zero)
    df['share_age_0_5'] = df['enrolment_age_0_5'] / df['total_enrolment'].replace(0, np.nan)
    df['share_age_5_17'] = df['enrolment_age_5_17'] / df['total_enrolment'].replace(0, np.nan)
    df['share_age_18_plus'] = df['enrolment_age_18_plus'] / df['total_enrolment'].replace(0, np.nan)
    
    # Fill NaN shares with 0
    cols = ['share_age_0_5', 'share_age_5_17', 'share_age_18_plus']
    df[cols] = df[cols].fillna(0)
    
    return df

def create_demographic_features(df):
    """
    Creates derived features for Demographic Update data.
    """
    df = df.copy()
    
    # Total Demo Updates
    df['total_demo_updates'] = df['demo_age_5_17'] + df['demo_age_18_plus']
    
    return df

def create_biometric_features(df):
    """
    Creates derived features for Biometric Update data.
    """
    df = df.copy()
    
    # Total Bio Updates
    df['total_bio_updates'] = df['bio_age_5_17'] + df['bio_age_18_plus']
    
    # Child vs Adult Bio Update Ratio (Proxy for Child-to-Adult transition intensity)
    # High bio_age_5_17 might indicate mandatory bio updates for children at age 5/15
    df['child_to_adult_bio_ratio'] = df['bio_age_5_17'] / df['bio_age_18_plus'].replace(0, 1)
    
    return df

def create_composite_features(enrol_df, demo_df, bio_df):
    """
    Merges datasets on Date, State, District (Pincode might be too granular for aggregate metrics, 
    but we can try. For Region Stress Index, filtering by District is better).
    """
    # Group by key columns to ensure unique rows before merge if needed, 
    # but the cleaned data should be daily records per pincode.
    # To correspond to 'Region', we usually mean District or State. 
    # Let's create features at the Pincode level first, as that's the base granularity.
    
    # Rename columns to avoid collision before merge
    e = enrol_df[['date', 'state', 'district', 'pincode', 'total_enrolment']].copy()
    d = demo_df[['date', 'state', 'district', 'pincode', 'total_demo_updates']].copy()
    b = bio_df[['date', 'state', 'district', 'pincode', 'total_bio_updates']].copy()
    
    # Merge
    # Outer merge to keep all records
    m = pd.merge(e, d, on=['date', 'state', 'district', 'pincode'], how='outer').fillna(0)
    m = pd.merge(m, b, on=['date', 'state', 'district', 'pincode'], how='outer').fillna(0)
    
    # Composite Features
    
    # 1. Total Activity Load
    m['total_activity'] = m['total_enrolment'] + m['total_demo_updates'] + m['total_bio_updates']
    
    # 2. Enrolment to Update Ratio (Enrolment / (Demo + Bio))
    m['enrolment_to_update_ratio'] = m['total_enrolment'] / (m['total_demo_updates'] + m['total_bio_updates']).replace(0, np.nan)
    
    # 3. Region Stress Index (Normalized Load)
    # Simple proxy: Activity per capita? We don't have population data.
    # Just raw load for now. We can normalize later.
    
    return m

def calculate_performance_score(df, enrol_weight=0.6, update_weight=0.4):
    """
    Calculates a weighted performance score.
    Higher enrolment is generally positive (coverage).
    High updates might be positive (maintenance) or negative (errors).
    Here we treat activity as 'Performance' in terms of throughput.
    """
    # Normalize columns to 0-1 scale first to make weights meaningful
    cols = ['total_enrolment', 'total_demo_updates', 'total_bio_updates']
    normalized_df = df.copy()
    
    for col in cols:
        max_val = df[col].max()
        if max_val > 0:
            normalized_df[col] = df[col] / max_val
        else:
            normalized_df[col] = 0
            
    # Score = w1 * Enrolment + w2 * (Demo + Bio)
    # Using provided weights
    score = (enrol_weight * normalized_df['total_enrolment']) + \
            (update_weight * (normalized_df['total_demo_updates'] + normalized_df['total_bio_updates']))
            
    return score * 100  # Scale to 0-100

def calculate_growth_rates(df, value_col, date_col='date', freq='M'):
    """
    Calculates Month-over-Month (MoM) growth rates.
    Aggregates data by date to calculate time-series growth.
    """
    # Aggregate by time
    ts = df.set_index(date_col).resample(freq)[value_col].sum()
    growth = ts.pct_change() * 100
    return growth
