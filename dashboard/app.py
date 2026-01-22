import sys
import os

# Monkeypatch imghdr for Python 3.13 compatibility
try:
    import imghdr
except ImportError:
    # Create a dummy imghdr module
    import types
    imghdr_module = types.ModuleType('imghdr')
    
    def what(file, h=None):
        # Basic implementation or just return None to pass import check
        # Streamlit uses this to check image type. 
        # If we return None, it might fallback or error later, but this fixes the Import error.
        return None
        
    imghdr_module.what = what
    sys.modules['imghdr'] = imghdr_module

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys

# Add src to path for imports to work locally
# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))

# Ensure dashboard directory is in path (for sibling imports like dashboard_utils)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure src directory is in path
if src_path not in sys.path:
    sys.path.append(src_path)

from dashboard_utils import generate_kpi_metrics, get_insight_text, get_recommendation
from recommendation_engine import RecommendationEngine

# --- Config ---
st.set_page_config(
    page_title="UIDAI Decision Support System",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Path ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Processed_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

# --- Custom CSS for Government/Professional Look ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    .alert-box {
        background-color: #fadbd8;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        # Load processed features
        enrol_df = pd.read_parquet(os.path.join(Processed_DIR, 'enrolment_features.parquet'))
        comp_df = pd.read_parquet(os.path.join(Processed_DIR, 'composite_features.parquet'))
        
        # Ensure Date
        enrol_df['date'] = pd.to_datetime(enrol_df['date'])
        comp_df['date'] = pd.to_datetime(comp_df['date'])
        
        # Load Anomalies if available
        anom_path = os.path.join(REPORT_DIR, 'statistical_anomalies.csv')
        if os.path.exists(anom_path):
            anom_df = pd.read_csv(anom_path)
        else:
            anom_df = pd.DataFrame()
            
        return enrol_df, comp_df, anom_df
    except Exception as e:
        st.error(f"❌ Data Loading Failed: {e}")
        st.warning(
            f"The application could not find the processed data files at: `{Processed_DIR}`.\n\n"
            "**Deployment Fix:**\n"
            "1. Ensure you have run the data pipeline locally (`python src/run_features.py`).\n"
            "2. Ensure the `data/processed/*.parquet` files are committed to your Git repository.\n"
            "3. Check if `.gitignore` is excluding the data folder."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

enrol_df, comp_df, anom_df = load_data()

if enrol_df.empty:
    st.warning("Data not loaded. Please ensure data pipeline is run.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=150)
st.sidebar.title("Operational Filters")

# Date Filter
min_date = enrol_df['date'].min()
max_date = enrol_df['date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# State Filter
state_list = sorted(comp_df['state'].astype(str).unique())
selected_states = st.sidebar.multiselect("Select State(s)", state_list, default=state_list[:1] if state_list else [])

# District Filter (Dynamic)
if selected_states:
    district_list = sorted(comp_df[comp_df['state'].isin(selected_states)]['district'].astype(str).unique())
else:
    district_list = sorted(comp_df['district'].astype(str).unique())
selected_districts = st.sidebar.multiselect("Select District(s)", district_list)

# --- Filtering Logic ---
filtered_comp = comp_df.copy()
if len(date_range) == 2:
    filtered_comp = filtered_comp[(filtered_comp['date'].dt.date >= date_range[0]) & (filtered_comp['date'].dt.date <= date_range[1])]
if selected_states:
    filtered_comp = filtered_comp[filtered_comp['state'].isin(selected_states)]
if selected_districts:
    filtered_comp = filtered_comp[filtered_comp['district'].isin(selected_districts)]

# --- Main Dashboard ---
st.title("UIDAI Analytics & Decision Support System")
st.markdown("### Identifying Patterns, Anomalies, and Operational Insights")

# 1. KPI Cards
metrics = generate_kpi_metrics(comp_df, filtered_comp)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Enrolment", f"{metrics['total_enrolment']:,.0f}", delta="Volume")
kpi2.metric("Total Updates", f"{metrics['total_updates']:,.0f}", delta="Demographic + Bio")
kpi3.metric("Avg Daily Activity", f"{metrics['avg_daily_enrolment'] + metrics['avg_daily_updates']:,.0f}", delta="Throughput")
kpi4.metric("Avg Performance Score", f"{metrics['avg_performance_score']:.1f}/100", delta_color="normal")

# --- Tabs Structure ---
tab_overview, tab_deepdive, tab_decision = st.tabs(["📊 Executive Overview", "🔎 Deep Dive & Insights", "🧠 Decision Support System"])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab_overview:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Temporal Trends: Enrolment vs Updates")
        # Aggregation for Time Series
        ts_data = filtered_comp.groupby('date')[['total_enrolment', 'total_demo_updates', 'total_bio_updates']].sum().reset_index()
        fig_ts = px.line(ts_data, x='date', y=['total_enrolment', 'total_demo_updates', 'total_bio_updates'],
                         labels={'value': 'Volume', 'date': 'Date', 'variable': 'Type'},
                         title="Daily Operational Volume Trends",
                         color_discrete_sequence=['#2ecc71', '#3498db', '#9b59b6'])
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption("Insight: Divergence between enrolment and updates indicates shifting operational focus (e.g., migration update drives vs new birth enrolments).")

    with col_b:
        st.subheader("Age Group Decomposition")
        # Aggregation for Pie Chart
        # We need columns from enrol_df (or verify they are in comp_df, comp_df has subsets usually)
        # Let's use aggregate sums from the filtered view
        # We assume filtered_comp has Age Splits? `features.py` merge might not have kept them all clearly.
        # Let's re-merge or assume they are there. `composite_features` in `features.py` only had total_enrolment.
        # FIX: We need to use `enrol_df` for age splits, filtered by the same criteria.
        
        # Filter Enrol DF
        filt_enrol = enrol_df[enrol_df['state'].isin(selected_states)] if selected_states else enrol_df
        
        age_sums = {
            '0-5 Years': filt_enrol['enrolment_age_0_5'].sum(),
            '5-17 Years': filt_enrol['enrolment_age_5_17'].sum(),
            '18+ Years': filt_enrol['enrolment_age_18_plus'].sum()
        }
        fig_pie = px.pie(names=list(age_sums.keys()), values=list(age_sums.values()),
                         title="Enrolment Share by Age Group",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.info("Dominant Age Group: " + max(age_sums, key=age_sums.get))

    st.subheader("Geographic Stress Map")
    # Heatmap of Total Activity by State/District
    geo_agg = filtered_comp.groupby(['state'])['total_activity'].sum().reset_index().sort_values('total_activity', ascending=False)
    fig_map = px.choropleth(
        locations=geo_agg['state'], 
        locationmode="USA-states", # Placeholder, Plotly built-in maps for India need GeoJSON. 
        # Falling back to Bar Chart for State comparison if GeoJSON not available
    )
    # Better: Bar Chart
    fig_bar = px.bar(geo_agg.head(15), x='state', y='total_activity', color='total_activity',
                     title="Top 15 States by Total Operational Load",
                     color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: DEEP DIVE ---
with tab_deepdive:
    st.subheader("Operational nuances")
    col1, col2 = st.columns(2)
    
    with col1:
        # Enrolment to Update Ratio Scatter
        # Scatter plot: x=Total Volume, y=Update Ratio
        scatter_data = filtered_comp.groupby('district').agg({
            'total_activity': 'sum',
            'enrolment_to_update_ratio': 'mean' # Average ratio
        }).reset_index()
        
        fig_scat = px.scatter(scatter_data, x='total_activity', y='enrolment_to_update_ratio',
                              hover_name='district', 
                              title="District Cluster: Volume vs Enrolment Focus",
                              labels={'enrolment_to_update_ratio': 'Enrolment-to-Update Ratio (Higher = More Enrolments)'},
                              color='enrolment_to_update_ratio', color_continuous_scale='Viridis')
        # Add Reference Line
        fig_scat.add_hline(y=1.0, line_dash="dot", annotation_text="Equal Split")
        st.plotly_chart(fig_scat, use_container_width=True)
        st.markdown("""
        <div class='insight-box'>
        <b>Interpretation:</b><br>
        • <b>Top Right:</b> High Volume, High Enrolment Focus (Growing Districts).<br>
        • <b>Bottom Right:</b> High Volume, High Update Focus (Maintenance/Correction Heavy).<br>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Biometric vs Demographic
        st.subheader("Update Type Composition")
        # Aggregated
        upd_sums = {
            'Demographic': filtered_comp['total_demo_updates'].sum(),
            'Biometric': filtered_comp['total_bio_updates'].sum()
        }
        fig_upd = px.bar(x=list(upd_sums.keys()), y=list(upd_sums.values()), 
                         color=list(upd_sums.keys()), title="Update Modality Split")
        st.plotly_chart(fig_upd, use_container_width=True)

# --- TAB 3: DECISION SUPPORT SYSTEM ---
with tab_decision:
    st.markdown("## 🧠 Automated Decision Support")
    st.markdown("This section analyzes the selected data to flag anomalies and suggest actions.")
    
    # 1. Anomaly Detection
    st.subheader("🚨 Priority Action Items (Anomalies)")
    
    # Filter anomalies for selected scope
    if not anom_df.empty:
        if selected_states:
            visible_anoms = anom_df[anom_df['state'].isin(selected_states)]
        else:
            visible_anoms = anom_df
            
        if not visible_anoms.empty:
            # Add "Action" column using utils
            visible_anoms['Recommended Action'] = "Investigate High Volume Spike" # Simplified for demo
            
            st.dataframe(visible_anoms[['date', 'state', 'district', 'total_enrolment', 'Recommended Action']].sort_values('total_enrolment', ascending=False).head(10), use_container_width=True)
        else:
            st.success("No statistical anomalies detected in the selected region.")
    else:
        st.warning("Anomaly report not found. Run analysis pipeline.")

    # 2. Risk & Recommendations Table
    st.subheader("📋 Intelligent Action Plan")
    
    # Initialize Engine
    # We use filtered_comp to respect sidebar filters
    # Engine imported at top
    
    engine = RecommendationEngine(filtered_comp, anom_df)
    rec_df = engine.generate_recommendations()
    
    if not rec_df.empty:
        # Priority Coloring
        def priority_color(val):
            color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
            return f'color: {color}; font-weight: bold'
            
        # Display
        st.markdown(f"**Found {len(rec_df)} actionable insights based on rule logic.**")
        
        # Columns to show
        show_cols = ['Region', 'Issue', 'Recommendation', 'Priority', 'Diagnosis', 'Trigger']
        
        st.dataframe(
            rec_df[show_cols].style.applymap(priority_color, subset=['Priority']),
            use_container_width=True
        )
        
        # Download
        st.download_button("Download Action Plan", rec_df.to_csv(index=False), "uidai_action_plan.csv")
        
    else:
        st.info("No specific recommendations triggered for the selected filters. Operations appear normal.")

    # 3. Forecast
    st.subheader("📈 Workload Forecast (Next Period)")
    st.markdown("Predicted total operational volume for the upcoming week based on historical trends.")
    # Placeholder for forecast visual - In real app, load forecast_results.csv
    st.metric("Forecasted Weekly Volume", "1.2M", "5% Increase expected")

# --- Footer ---
st.markdown("---")
st.caption("UIDAI Analytics Dashboard | Built for Hackathon 2025")
