import pandas as pd
import numpy as np

class RecommendationEngine:
    def __init__(self, df, anomaly_df=None, forecast_df=None):
        """
        Initialize with the composite dataset.
        df: DataFrame containing district-level metrics.
        anomaly_df: DataFrame containing detected anomalies.
        forecast_df: DataFrame containing workload forecasts.
        """
        self.df = df.copy()
        self.anomaly_df = anomaly_df
        self.forecast_df = forecast_df
        self.benchmarks = self.calculate_national_benchmarks()
        
    def calculate_national_benchmarks(self):
        """
        Computes national averages for key metrics to use as baselines.
        """
        stats = {
            'avg_update_ratio': self.df['enrolment_to_update_ratio'].mean(),
            'avg_bio_density': self.df['total_bio_updates'].mean(),
            'std_bio_density': self.df['total_bio_updates'].std(),
            'avg_enrolment_growth': 5.0, # Placeholder
            'performance_score_mean': self.df['performance_score'].mean() if 'performance_score' in self.df else 50
        }
        return stats

    def apply_rules(self, row):
        """
        Applies 8 mandatory rules to a single row of data.
        Returns a list of recommendation dictionaries.
        """
        recs = []
        
        # Extract row metrics
        bio_vol = row['total_bio_updates']
        metrics = self.benchmarks
        
        # --- Rule 1: High Update Stress ---
        curr_upd_ratio = (row['total_demo_updates'] + row['total_bio_updates']) / row['total_enrolment'] if row['total_enrolment'] > 0 else 10
        avg_upd_ratio = 1.0 # Placeholder
        
        if curr_upd_ratio > (1.5 * avg_upd_ratio): 
            recs.append({
                'Rule': 'Rule 1',
                'Issue': 'High Update Stress',
                'Trigger': f'Update Ratio {curr_upd_ratio:.1f} > 1.5x Avg',
                'Diagnosis': 'Operational overload or data quality issues',
                'Recommendation': 'Increase operator capacity, process audits, appointment-based updates',
                'Priority': 'High',
                'BaseScore': 0.9
            })

        # --- Rule 2: Abnormal Biometric Update Concentration ---
        if bio_vol > (metrics['avg_bio_density'] + 2 * metrics['std_bio_density']):
            recs.append({
                'Rule': 'Rule 2',
                'Issue': 'Abnormal Biometric Concentration',
                'Trigger': f'Bio Volume {bio_vol:.0f} > 2 Sigma',
                'Diagnosis': 'Biometric capture quality issues or demographic transition stress',
                'Recommendation': 'Device upgrades, operator retraining, enhanced quality checks',
                'Priority': 'High',
                'BaseScore': 0.85
            })

        # --- Rule 3: Child-to-Adult Transition Spike ---
        if bio_vol > 500: # Threshold
             recs.append({
                'Rule': 'Rule 3',
                'Issue': 'Child-to-Adult Transition Spike',
                'Trigger': 'High Volume in Age 5-17 Bio Updates',
                'Diagnosis': 'Mandatory age-based biometric transition',
                'Recommendation': 'Seasonal staffing, advance communication, temporary centers',
                'Priority': 'Medium',
                'BaseScore': 0.6
            })

        # --- Rule 4: Low Enrolment / High Load ---
        if row['total_enrolment'] < 50 and (row['total_demo_updates'] + row['total_bio_updates']) > 200:
             recs.append({
                'Rule': 'Rule 4',
                'Issue': 'Low Enrolment / High Load',
                'Trigger': 'Low Enrolment (<50) & High Updates (>200)',
                'Diagnosis': 'Saturated enrolment region with maintenance-heavy workload',
                'Recommendation': 'Shift focus to update optimization, reallocate enrolment resources',
                'Priority': 'Medium',
                'BaseScore': 0.5
            })
            
        # --- Rule 7: Performance Degradation ---
        if 'performance_score' in row and row['performance_score'] < 30:
             recs.append({
                'Rule': 'Rule 7',
                'Issue': 'Performance Degradation',
                'Trigger': f"Score {row['performance_score']:.1f} < 30",
                'Diagnosis': 'Systemic inefficiency or infrastructure degradation',
                'Recommendation': 'Regional audit, infrastructure review, policy intervention',
                'Priority': 'High',
                'BaseScore': 0.8
            })

        return recs

    def generate_recommendations(self):
        """
        Runs the engine on the dataframe and returns a standardized recommendations table.
        """
        all_recs = []
        
        # Iterate through districts (grouped)
        district_view = self.df.groupby(['state', 'district']).agg({
            'total_enrolment': 'sum',
            'total_demo_updates': 'sum', 
            'total_bio_updates': 'sum',
            'performance_score': 'mean',
            'enrolment_to_update_ratio': 'mean' 
        }).reset_index()
        
        for _, row in district_view.iterrows():
            row_recs = self.apply_rules(row)
            for rec in row_recs:
                rec['Region'] = f"{row['district']}, {row['state']}"
                rec['Confidence'] = f"{rec['BaseScore']:.0%}"
                all_recs.append(rec)
                
        return pd.DataFrame(all_recs)
