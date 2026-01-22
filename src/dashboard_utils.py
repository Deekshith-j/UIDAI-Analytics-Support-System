import pandas as pd
import numpy as np

def generate_kpi_metrics(df, filtered_df):
    """
    Calculates high-level KPIs for the dashboard.
    Returns a dict of metrics.
    """
    metrics = {}
    
    # Total Volumes
    metrics['total_enrolment'] = filtered_df['total_enrolment'].sum()
    metrics['total_updates'] = filtered_df['total_demo_updates'].sum() + filtered_df['total_bio_updates'].sum()
    
    # Growth Rates (Simple comparison with previous period if available, else just current total)
    # Ideally we'd calculate MoM growth here if time data permits.
    # Let's calculate average daily volume as a proxy for "Run Rate"
    days = filtered_df['date'].nunique()
    if days > 0:
        metrics['avg_daily_enrolment'] = metrics['total_enrolment'] / days
        metrics['avg_daily_updates'] = metrics['total_updates'] / days
    else:
        metrics['avg_daily_enrolment'] = 0
        metrics['avg_daily_updates'] = 0
        
    # Anomaly/Risk Count
    # Assuming 'performance_score' exists and low score means risk, or we use a threshold on stress index
    # Let's use the 'is_high_risk' from models if available, or simpler threshold
    # For now, let's look at average performance score
    if 'performance_score' in filtered_df.columns:
        metrics['avg_performance_score'] = filtered_df['performance_score'].mean()
    else:
        metrics['avg_performance_score'] = 0
        
    return metrics

def get_insight_text(row):
    """
    Generates a natural language insight for a specific row (District/State summary).
    """
    insights = []
    
    # 1. Enrolment vs Updates
    total_activity = row['total_enrolment'] + row['total_demo_updates'] + row['total_bio_updates']
    if total_activity > 0:
        update_ratio = (row['total_demo_updates'] + row['total_bio_updates']) / total_activity
        if update_ratio > 0.7:
            insights.append(f"High update intensity ({update_ratio:.1%}).")
        elif update_ratio < 0.1:
            insights.append(f"Dominant fresh enrolment ({1-update_ratio:.1%}).")
            
    # 2. Child Nuance
    if 'enrolment_age_0_5' in row and row['total_enrolment'] > 0:
        child_share = row['enrolment_age_0_5'] / row['total_enrolment']
        if child_share > 0.4:
            insights.append(f"High child enrolment share ({child_share:.1%}).")
            
    # 3. Biometric
    if 'child_to_adult_bio_ratio' in row and row['child_to_adult_bio_ratio'] > 2:
         insights.append("Potential localized child bio-update drive.")
         
    return " ".join(insights) if insights else "Activity within normal parameters."

def get_recommendation(row):
    """
    Returns a prescriptive action based on metrics.
    """
    recs = []
    
    # Check Stress/Load
    # We can use 'performance_score' (higher is better?) or 'total_activity'
    # Assuming we have access to some standardized 'stress_index' or just using raw load
    # Let's use a heuristic based on 'enrolment_to_update_ratio' and volumes
    
    if row.get('enrolment_to_update_ratio', 0) < 0.2:
        recs.append("Deploy Mobile Update Units (High Update Demand).")
        
    if row.get('child_to_adult_bio_ratio', 0) > 1.5:
        recs.append("Conduct School Camps (High Child Biometric Load).")
        
    if row.get('performance_score', 100) < 40: # Assuming 0-100 scale
        recs.append("Audit Center Operations (Low Performance Score).")
        
    if not recs:
        recs.append("Monitor for standard operations.")
        
    return " | ".join(recs)

def calculate_trend_delta(current_val, prev_val):
    """
    Calculates percentage change.
    """
    if prev_val == 0:
        return 0
    return ((current_val - prev_val) / prev_val) * 100
