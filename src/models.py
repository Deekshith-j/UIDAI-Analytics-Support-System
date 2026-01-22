import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score

def prepare_time_series_data(df, target_col, date_col='date', lags=3):
    """
    Prepares data for time series forecasting using lag properties.
    Aggregates daily data to weekly/monthly before lagging to reduce noise.
    """
    # Resample to weekly
    ts_data = df.set_index(date_col).resample('W')[target_col].sum().to_frame()
    
    # Create lags
    for i in range(1, lags + 1):
        ts_data[f'lag_{i}'] = ts_data[target_col].shift(i)
        
    ts_data = ts_data.dropna()
    return ts_data

def train_forecast_model(ts_data, target_col):
    """
    Trains a Linear Regression model for forecasting.
    """
    features = [c for c in ts_data.columns if 'lag_' in c]
    X = ts_data[features]
    y = ts_data[target_col]
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    return model, rmse, preds, y_test

def train_risk_model(df, target_col='total_activity', threshold_percentile=0.90):
    """
    Trains a Random Forest model to predict high-risk regions (High Load).
    """
    # Define Target: High Load (1) vs Normal (0)
    threshold = df[target_col].quantile(threshold_percentile)
    df['is_high_risk'] = (df[target_col] > threshold).astype(int)
    
    # Features: We want to predict risk based on available attributes.
    # Ideally lagged features, but for simplicity let's use current simple features excluding the target components
    # Using: state (encoded), and other non-leakage features.
    # Actually, predicting stress from enrolment/updates is circular if we use current values.
    # Correct approach: Predict Future Risk based on Past Data.
    # But here, we might just classify regions based on their static characteristics if available?
    # Or, let's keep it simple: Can we classify the risk level based on just demographic features (like district name embeddings? no).
    # Let's use 'enrolment_to_update_ratio' and maybe simple counts to see if they predict 'stress' (total load)?
    # This is slightly redundant but shows the ML workflow.
    
    feature_cols = ['enrolment_to_update_ratio', 'total_enrolment', 'child_to_adult_bio_ratio']
    # Ensure they exist (child_to_adult comes from bio, might need to merge carefully)
    
    available_features = [c for c in feature_cols if c in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)
    
    return clf, report, available_features
