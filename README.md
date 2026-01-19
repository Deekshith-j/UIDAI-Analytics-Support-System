# UIDAI Analytics & Decision Support System

## 1. Problem Statement
The goal of this project is to analyze anonymised Aadhaar Enrolment, Demographic, and Biometric update datasets to uncover operational patterns, detect anomalies (fraud/inefficiency), and forecast future workload. This system supports data-driven decision-making for resource allocation and policy formulation.

## 2. Methodology
The solution is built using a modular Python architecture:
- **Data Ingestion**: Automated loading and merging of split CSV files.
- **Cleaning**: Deduplication and rigorous state name standardization using heuristic mapping.
- **Feature Engineering**: Creation of derived metrics like *Enrolment-to-Update Ratio*, *Child-to-Adult Biometric Transition*, and a composite *Region Stress Index*.
- **EDA**: Visual analysis of trends, seasonality, and regional disparities.
- **Analytics**:
    - **Anomaly Detection**: Statistical (Z-score) and ML-based (Isolation Forest) methods.
    - **Predictive Modeling**: Time-series forecasting and Risk classification.
- **Decision Support Dashboard**:
    - **Executive Overview**: High-level trends and age-group demographics.
    - **Deep Dive**: Analysis of Update Modalities and Enrolment-to-Update ratios.
    - **Automated Insights**: Rule-based engine generating natural language alerts (e.g., "High Child Biometric Load").
    - **Recommendations**: Actionable operational suggestions (e.g., "Deploy Mobile Vans") based on data patterns.

## 3. Key Insights
- **Enrolment Trends**: Monthly trends reveal seasonal peaks (visualized in reports).
- **Regional Disparities**: Certain states show disproportionately high update ratios vs enrolments, suggesting migration or data correction drives.
- **Anomalies**: Outlier districts identified with <1% probability scores indicate potential fraudulent activity or system errors.
- **Predictive Capability**: The system forecasts next-month workload with a measured RMSE, enabling proactive staffing.

## 4. Impact for UIDAI
- **Operational Efficiency**: Predicts stress regions to optimize center allocation.
- **Fraud Detection**: Flags aggregated anomalies for ground-level investigation.
- **Policy**: "Child-to-Adult" transition metrics help target mandatory biometric update campaigns.

## 5. Project Structure
- `data/`: Raw and processed datasets (parquet).
- `src/`: Source code for loading, cleaning, features, models.
    - `data_loader.py`: Ingestion logic.
    - `cleaning.py`: Standardization pipelines.
    - `features.py`: Metric calculation.
    - `models.py`: ML and Forecasting.
- `notebooks/`: Jupyter notebooks for exploration.
- `dashboard/`: Streamlit application (`app.py`).
- `reports/`: Generated PDF/CSV reports and plots.

## 6. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run Pipeline (Optional, data already processed):
   - `python src/run_cleaning.py`
   - `python src/run_features.py`
   - `python src/run_models.py`
3. Launch Dashboard:
   - `python -m streamlit run dashboard/app.py`

## 7. Verification & Testing
To verify the Recommendation Engine logic:
- **Rule 1 (Stress)**: Filter dashboard to a district with High Updates. Check for "High Update Stress" alert.
- **Rule 2 (Biometric)**: Look for districts with anomalously high biometric counts (> 2 std dev).
- **Rule 3 (Child Transition)**: Filter for districts with >500 biometric updates to see the "Child-to-Adult" trigger.
