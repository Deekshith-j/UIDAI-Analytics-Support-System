import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_time_series(df, date_col, value_col, title, output_path):
    """
    Plots a time series of a given metric.
    """
    plt.figure()
    # Resample to monthly if dense
    if 'date' in df.columns:
        ts = df.set_index('date').resample('M')[value_col].sum()
    else:
        ts = df
        
    ts.plot(kind='line', linewidth=2.5, marker='o')
    plt.title(title, fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_state_comparison(df, value_col, title, output_path, top_n=10):
    """
    Plots the top N states by a specific metric.
    """
    plt.figure(figsize=(14, 8))
    state_agg = df.groupby('state')[value_col].sum().sort_values(ascending=False).head(top_n)
    
    sns.barplot(x=state_agg.values, y=state_agg.index, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel("Count")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_age_distribution(df, age_cols, title, output_path):
    """
    Plots age distribution from multiple columns.
    """
    plt.figure()
    sums = df[age_cols].sum()
    sums.index = [col.replace('enrolment_age_', '').replace('demo_age_', '').replace('bio_age_', '') for col in sums.index]
    
    plt.pie(sums, labels=sums.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(df, x_col, y_col, value_col, title, output_path):
    """
    Plots a heatmap (e.g., Year vs Month for seasonality).
    """
    plt.figure()
    pivot = df.pivot_table(index=df[x_col].dt.year, columns=df[x_col].dt.month, values=value_col, aggfunc='sum')
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
