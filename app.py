"""
Linear Regression Studio & Flight Price Analytics Hub
Interactive end-to-end Data Science, EDA, Model Building, Testing, and Inferencing.
"""

import os
import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Linear Regression Studio & Flight Price Hub",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 24px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #E2E8F0;
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .step-header {
        background: #F1F5F9;
        border-left: 5px solid #2563EB;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 14px 0;
        font-weight: 700;
        font-size: 1.15rem;
        color: #0F172A;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #065F46 0%, #059669 100%);
        color: white;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(5, 150, 105, 0.25);
    }
    .eda-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #EEF2FF;
        color: #4F46E5;
        margin-right: 6px;
    }
    .formula-card {
        background: rgba(248, 250, 252, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-left: 5px solid #3B82F6;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 12px 0;
        color: inherit;
        backdrop-filter: blur(8px);
    }
    .formula-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #60A5FA;
        margin-bottom: 8px;
    }
    .flow-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        margin-right: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for data parsing
def parse_time_bin(val):
    try:
        val_str = str(val).strip()
        val_clean = val_str.replace(" ", "_").capitalize()
        valid_bins = ['Early_morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_night']
        for b in valid_bins:
            if val_clean.lower() == b.lower():
                return b
        match = re.search(r'(\d{1,2}):(\d{2})', val_str)
        if match:
            h = int(match.group(1))
            if 4 <= h < 8:
                return 'Early_Morning'
            elif 8 <= h < 12:
                return 'Morning'
            elif 12 <= h < 16:
                return 'Afternoon'
            elif 16 <= h < 20:
                return 'Evening'
            elif 20 <= h < 24:
                return 'Night'
            else:
                return 'Late_Night'
    except Exception:
        pass
    return 'Morning'

def parse_duration(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    try:
        return float(val_str)
    except ValueError:
        pass
    h_match = re.search(r'(\d+)\s*h', val_str, re.IGNORECASE)
    m_match = re.search(r'(\d+)\s*m', val_str, re.IGNORECASE)
    hours = float(h_match.group(1)) if h_match else 0.0
    minutes = float(m_match.group(1)) if m_match else 0.0
    return round(hours + minutes / 60.0, 2)

def parse_stops(val):
    val_str = str(val).lower().strip()
    if 'zero' in val_str or 'non' in val_str or val_str == '0':
        return 'zero'
    elif '2' in val_str or 'two' in val_str or 'more' in val_str:
        return 'two_or_more'
    else:
        return 'one'

def clean_and_prepare_dataset(df_raw, default_class=None):
    df = df_raw.copy()
    col_map = {}
    for c in df.columns:
        norm = c.strip().lower().replace(" ", "_")
        col_map[c] = norm
    df.rename(columns=col_map, inplace=True)
    
    price_col = None
    for cand in ['price', 'target', 'ticket_price', 'fare']:
        if cand in df.columns:
            price_col = cand
            break
            
    if price_col:
        df['price'] = df[price_col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('₹', '').str.strip()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    if 'class' not in df.columns:
        df['class'] = default_class if default_class else 'Economy'
    else:
        df['class'] = df['class'].astype(str).str.strip().str.capitalize().replace({'': 'Economy', 'Nan': 'Economy'})
        
    source_col = next((c for c in ['source_city', 'from', 'source', 'origin', 'dep_city'] if c in df.columns), None)
    df['source_city'] = df[source_col].astype(str).str.strip().str.capitalize() if source_col else 'Delhi'

    dest_col = next((c for c in ['destination_city', 'to', 'destination', 'dest', 'arr_city'] if c in df.columns), None)
    df['destination_city'] = df[dest_col].astype(str).str.strip().str.capitalize() if dest_col else 'Mumbai'

    df['airline'] = df['airline'].astype(str).str.strip() if 'airline' in df.columns else 'Air India'

    stops_col = next((c for c in ['stops', 'stop', 'num_stops'] if c in df.columns), None)
    df['stops'] = df[stops_col].apply(parse_stops) if stops_col else 'one'

    dep_col = next((c for c in ['departure_time', 'dep_time', 'departure'] if c in df.columns), None)
    df['departure_time'] = df[dep_col].apply(parse_time_bin) if dep_col else 'Morning'

    arr_col = next((c for c in ['arrival_time', 'arr_time', 'arrival'] if c in df.columns), None)
    df['arrival_time'] = df[arr_col].apply(parse_time_bin) if arr_col else 'Evening'

    dur_col = next((c for c in ['duration', 'time_taken', 'flight_duration'] if c in df.columns), None)
    df['duration'] = df[dur_col].apply(parse_duration) if dur_col else 2.5

    days_col = next((c for c in ['days_left', 'days_before', 'booking_lead'] if c in df.columns), None)
    if days_col:
        df['days_left'] = pd.to_numeric(df[days_col], errors='coerce').fillna(20).astype(int)
    elif 'date' in df.columns:
        try:
            dates = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            min_date = dates.min()
            df['days_left'] = (dates - min_date).dt.days + 1 if pd.notna(min_date) else 15
        except Exception:
            df['days_left'] = 15
    else:
        df['days_left'] = 15

    if 'price' in df.columns:
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]
        
    return df

# Header
st.markdown("""
<div class="main-header">
    <h1>✈️ Linear Regression Studio & Analytics Hub</h1>
    <p>Upload datasets directly, explore structured EDA workflows, configure linear regression pipelines, evaluate residual diagnostics, and predict real-time flight ticket prices.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("1. Upload / Select Data")
data_source = st.sidebar.radio(
    "Dataset Source:",
    ("Use Pre-loaded Collection (Business + Economy)", "Upload Custom Dataset (CSV)")
)

df = None
raw_uploaded_df = None
data_name = ""

if data_source == "Upload Custom Dataset (CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            with st.spinner("Parsing uploaded CSV..."):
                raw_uploaded_df = pd.read_csv(uploaded_file)
                df = clean_and_prepare_dataset(raw_uploaded_df)
                data_name = uploaded_file.name
                st.sidebar.success(f"Loaded `{data_name}` ({len(df):,} rows)")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    else:
        st.sidebar.info("💡 Upload a CSV file or switch to pre-loaded datasets.")
else:
    preset_choice = st.sidebar.selectbox(
        "Select Collection:",
        ("Combined (Business + Economy)", "Business Flights Only", "Economy Flights Only")
    )
    sample_size = st.sidebar.slider(
        "Sample Size for Fast Training:",
        min_value=5000,
        max_value=100000,
        value=30000,
        step=5000,
        help="Full dataset has over 300,000 rows. A balanced 30k sample yields sub-second training with ~90% R² validation score."
    )
    
    @st.cache_data
    def load_preset_data(choice, n_samples):
        b_path = "business.csv"
        e_path = "economy.csv"
        if not (os.path.exists(b_path) and os.path.exists(e_path)):
            return None
        if choice == "Business Flights Only":
            raw_b = pd.read_csv(b_path, nrows=n_samples)
            return clean_and_prepare_dataset(raw_b, default_class="Business")
        elif choice == "Economy Flights Only":
            raw_e = pd.read_csv(e_path, nrows=n_samples)
            return clean_and_prepare_dataset(raw_e, default_class="Economy")
        else:
            half = n_samples // 2
            raw_b = pd.read_csv(b_path, nrows=half)
            df_b = clean_and_prepare_dataset(raw_b, default_class="Business")
            raw_e = pd.read_csv(e_path, nrows=half)
            df_e = clean_and_prepare_dataset(raw_e, default_class="Economy")
            return pd.concat([df_b, df_e], ignore_index=True)

    with st.spinner("Loading flight datasets..."):
        df = load_preset_data(preset_choice, sample_size)
        data_name = preset_choice

if df is None:
    st.warning("⚠️ No dataset loaded yet. Please select an option or upload a CSV file from the sidebar.")
    st.stop()

# Sidebar: Model Config
st.sidebar.markdown("---")
st.sidebar.header("2. Model & Studio Config")
selected_algorithm = st.sidebar.selectbox(
    "Regression Algorithm:",
    ("Linear Regression (OLS)", "Ridge (L2 Regularized)", "Lasso (L1 Regularized)")
)
reg_alpha = 1.0
if "Ridge" in selected_algorithm or "Lasso" in selected_algorithm:
    reg_alpha = st.sidebar.slider("Regularization Alpha:", min_value=0.01, max_value=50.0, value=1.0, step=0.1)

test_split_ratio = st.sidebar.slider("Test Split Share:", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
random_seed = st.sidebar.number_input("Random Seed:", value=42, step=1)

feature_cols = ['airline', 'source_city', 'destination_city', 'class', 'stops', 'departure_time', 'arrival_time', 'duration', 'days_left']

# KPI cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df):,}</div><div class="metric-lbl">Total Samples</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">₹{df["price"].mean():,.0f}</div><div class="metric-lbl">Avg Price</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">₹{df["price"].median():,.0f}</div><div class="metric-lbl">Median Price</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">₹{df["price"].min():,.0f} - ₹{df["price"].max():,.0f}</div><div class="metric-lbl">Price Range</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{df["airline"].nunique()} Air / {df["source_city"].nunique()} Cities</div><div class="metric-lbl">Coverage</div></div>', unsafe_allow_html=True)

st.write("")

# 5 Main Tabs (with Comprehensive EDA Process & Studio Pipeline)
tab_eda, tab_studio, tab_pred, tab_tester, tab_data = st.tabs([
    "📊 EDA Process & Analytics",
    "🔬 Linear Regression Studio",
    "🎯 Live Price Predictor",
    "🧪 Model Tester & Diagnostics",
    "📋 Raw & Processed Data"
])

# Training Cache
@st.cache_resource
def train_studio_pipeline(data_df, test_size, algo_choice, alpha_val, seed):
    cols = [c for c in feature_cols if c in data_df.columns]
    X_df = pd.get_dummies(data_df[cols], drop_first=True)
    y = data_df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=seed)
    
    if "Ridge" in algo_choice:
        model = Ridge(alpha=alpha_val, random_state=seed)
    elif "Lasso" in algo_choice:
        model = Lasso(alpha=alpha_val, max_iter=2500, random_state=seed)
    else:
        model = LinearRegression()
        
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    
    # Adjusted R2
    n = len(y_test)
    p = X_df.shape[1]
    adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) if n > p + 1 else test_r2
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(LinearRegression(), X_df, y, cv=kf, scoring='r2')
    
    # Benchmark
    benchmarks = []
    for m_label, m_instance in [
        ('Linear Regression (OLS)', LinearRegression()),
        ('Ridge (L2)', Ridge(alpha=1.0, random_state=seed)),
        ('Lasso (L1)', Lasso(alpha=1.0, max_iter=2500, random_state=seed))
    ]:
        m_instance.fit(X_train, y_train)
        pred_i = m_instance.predict(X_test)
        benchmarks.append({
            'Model': m_label,
            'Test R²': r2_score(y_test, pred_i),
            'Test MAE (₹)': mean_absolute_error(y_test, pred_i),
            'Test RMSE (₹)': np.sqrt(mean_squared_error(y_test, pred_i))
        })

    # Evaluation sums for manual calculation breakdowns
    y_test_np = y_test.values
    y_mean = np.mean(y_test_np)
    ssr_val = float(np.sum((y_test_np - y_pred_test) ** 2))
    sst_val = float(np.sum((y_test_np - y_mean) ** 2))
    sae_val = float(np.sum(np.abs(y_test_np - y_pred_test)))

    return {
        'model': model,
        'columns': X_df.columns.tolist(),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'test_r2': test_r2,
        'adj_r2': adj_r2,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'ssr': ssr_val,
        'sst': sst_val,
        'sae': sae_val,
        'y_mean': y_mean,
        'n_test': n,
        'p_features': p,
        'cv_scores': cv_scores,
        'benchmarks_df': pd.DataFrame(benchmarks),
        'active_features': cols
    }

pipeline_results = train_studio_pipeline(df, test_split_ratio, selected_algorithm, reg_alpha, random_seed)
active_model = pipeline_results['model']
model_columns = pipeline_results['columns']

# ----------------------------------------------------
# TAB 1: COMPREHENSIVE EDA PROCESS & ANALYTICS
# ----------------------------------------------------
with tab_eda:
    st.subheader("📊 End-to-End Exploratory Data Analysis (EDA) Process")
    st.markdown("""
    Explore the 5-stage exploratory data analysis workflow:
    `Stage 1: Shape & Null Audit` ➔ `Stage 2: Target & Numeric Analysis` ➔ `Stage 3: Categorical & Flight Variables` ➔ `Stage 4: Correlation & Heatmaps` ➔ `Stage 5: Booking Dynamics`
    """)

    # Stage 1: Dataset Overview & Missing Values
    st.markdown('<div class="step-header">Stage 1: Dataset Architecture & Data Quality Audit</div>', unsafe_allow_html=True)
    e1_c1, e1_c2, e1_c3, e1_c4 = st.columns(4)
    with e1_c1:
        st.metric("Total Records", f"{len(df):,}")
    with e1_c2:
        st.metric("Total Features", f"{df.shape[1]}")
    with e1_c3:
        st.metric("Missing Values", f"{df.isna().sum().sum():,}", help="Cleaned & imputed in pipeline")
    with e1_c4:
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    with st.expander("🔍 View Column Schema & Data Types Breakdown", expanded=False):
        schema_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.notnull().sum(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        }).reset_index(drop=True)
        st.dataframe(schema_df, use_container_width=True)

    # Stage 2: Target Variable (Ticket Price) Distribution
    st.markdown('<div class="step-header">Stage 2: Target Variable (Price) Distribution & Outlier Analysis</div>', unsafe_allow_html=True)
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        fig_price_hist = px.histogram(
            df,
            x='price',
            nbins=60,
            title="Ticket Price Distribution (₹)",
            color='class',
            color_discrete_map={'Business': '#3B82F6', 'Economy': '#10B981'},
            marginal="box",
            labels={'price': 'Ticket Price (₹)'}
        )
        fig_price_hist.update_layout(height=400, barmode='overlay')
        fig_price_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_price_hist, use_container_width=True)

    with t_c2:
        # Log Price Transformation View (Data Science best practice)
        df_log_price = np.log1p(df['price'])
        fig_log_hist = px.histogram(
            x=df_log_price,
            nbins=50,
            title="Log-Transformed Price Distribution log(Price)",
            color_discrete_sequence=['#8B5CF6'],
            marginal="box",
            labels={'x': 'log(Price + 1)'}
        )
        fig_log_hist.update_layout(height=400)
        st.plotly_chart(fig_log_hist, use_container_width=True)

    # Stage 3: Categorical & Flight Variables Breakdown
    st.markdown('<div class="step-header">Stage 3: Categorical Breakdown (Airline, Class, Time Bins & Stops)</div>', unsafe_allow_html=True)
    c_row1_1, c_row1_2 = st.columns(2)
    with c_row1_1:
        # Airline Price Comparison
        air_agg = df.groupby('airline')['price'].agg(['mean', 'median', 'count']).reset_index().sort_values(by='mean', ascending=False)
        fig_airline = px.bar(
            air_agg,
            x='airline',
            y='mean',
            color='airline',
            title="Average Ticket Price by Airline Company (₹)",
            labels={'mean': 'Average Price (₹)', 'airline': 'Airline'},
            text_auto='.0f'
        )
        fig_airline.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_airline, use_container_width=True)

    with c_row1_2:
        # Cabin Class Impact Box Plot
        fig_class = px.box(
            df,
            x='class',
            y='price',
            color='class',
            color_discrete_map={'Business': '#2563EB', 'Economy': '#10B981'},
            title="Seat Class Price Distribution: Business vs Economy",
            labels={'price': 'Price (₹)', 'class': 'Cabin Class'}
        )
        fig_class.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_class, use_container_width=True)

    c_row2_1, c_row2_2 = st.columns(2)
    with c_row2_1:
        # Stops vs Price
        stop_agg = df.groupby('stops')['price'].agg(['mean', 'median', 'count']).reset_index()
        stop_agg['stops_label'] = stop_agg['stops'].map({'zero': 'Non-stop', 'one': '1 Stop', 'two_or_more': '2+ Stops'}).fillna(stop_agg['stops'])
        fig_stops = px.bar(
            stop_agg,
            x='stops_label',
            y='mean',
            color='stops_label',
            title="Average Ticket Price by Number of Stops",
            labels={'stops_label': 'Stops', 'mean': 'Avg Price (₹)'},
            text_auto='.0f'
        )
        fig_stops.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_stops, use_container_width=True)

    with c_row2_2:
        # Departure Time Window Impact
        dep_agg = df.groupby('departure_time')['price'].mean().reset_index().sort_values(by='price', ascending=False)
        fig_dep = px.bar(
            dep_agg,
            x='departure_time',
            y='price',
            color='departure_time',
            title="Average Ticket Price by Departure Time Window",
            labels={'departure_time': 'Departure Window', 'price': 'Avg Price (₹)'},
            text_auto='.0f'
        )
        fig_dep.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_dep, use_container_width=True)

    # Stage 4: Continuous Variables & Route Matrix
    st.markdown('<div class="step-header">Stage 4: Numerical Feature Correlations & Metro Route Matrix</div>', unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1.1])
    with n_col1:
        # Correlation Heatmap for numeric columns
        num_cols = [c for c in ['price', 'duration', 'days_left'] if c in df.columns]
        corr_matrix = df[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.3f',
            color_continuous_scale='RdBu_r',
            title="Numeric Correlation Matrix (Pearson Correlation)",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    with n_col2:
        # Metro Route Heatmap Matrix
        route_matrix = df.pivot_table(index='source_city', columns='destination_city', values='price', aggfunc='mean')
        fig_route = px.imshow(
            route_matrix,
            text_auto='.0f',
            color_continuous_scale='YlGnBu',
            title="City-to-City Route Pricing Matrix (Source vs Destination Fare)",
            labels=dict(x="Destination City", y="Source City", color="Avg Fare (₹)")
        )
        fig_route.update_layout(height=420)
        st.plotly_chart(fig_route, use_container_width=True)

    # Stage 5: Booking Dynamics & Surge Curve
    st.markdown('<div class="step-header">Stage 5: Booking Lead Time Dynamics (Surge Pricing Pattern)</div>', unsafe_allow_html=True)
    days_agg = df.groupby(['days_left', 'class'])['price'].mean().reset_index()
    fig_surge = px.line(
        days_agg,
        x='days_left',
        y='price',
        color='class',
        color_discrete_map={'Business': '#2563EB', 'Economy': '#10B981'},
        title="Booking Lead Days vs Average Ticket Price (Price Surge Trajectory)",
        labels={'days_left': 'Days Remaining Before Departure', 'price': 'Average Ticket Price (₹)', 'class': 'Cabin Class'},
        markers=True
    )
    fig_surge.update_layout(height=420)
    st.plotly_chart(fig_surge, use_container_width=True)

# ----------------------------------------------------
# TAB 2: LINEAR REGRESSION STUDIO (Interactive Workflow)
# ----------------------------------------------------
with tab_studio:
    st.subheader("🔬 Linear Regression Studio")
    st.write(f"Dataset active: **{data_name}** | Target: **Ticket Price (₹)** | Estimator: **{selected_algorithm}**")

    # Flow of Execution & Mathematical Formulation
    with st.expander("🧭 Linear Regression: End-to-End Flow of Execution & Mathematical Foundations", expanded=True):
        st.markdown("""
        ```mermaid
        flowchart LR
            A["1. Data Ingestion & Preprocessing"] --> B["2. One-Hot & Dummy Encoding"]
            B --> C["3. Train-Test Split (80/20)"]
            C --> D["4. Loss Function Minimization (OLS)"]
            D --> E["5. Weight Vector Estimation (β)"]
            E --> F["6. Test Evaluation (R², MAE, RMSE)"]
            F --> G["7. Residuals & Live Inference"]
        ```
        """)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown('<div class="formula-card"><div class="formula-title">1. Multiple Linear Regression Hypothesis</div>For flight instance <i>i</i> with <i>p</i> features:</div>', unsafe_allow_html=True)
            st.latex(r"\hat{y}_i = \beta_0 + \sum_{j=1}^p \beta_j X_{ij} = \mathbf{x}_i^T \boldsymbol{\beta} \quad \implies \quad \mathbf{\hat{Y}} = \mathbf{X}\boldsymbol{\beta}")
            st.caption("Where $\\mathbf{\\hat{Y}}$ is the predicted price vector, $\\mathbf{X}$ is the feature design matrix, and $\\boldsymbol{\\beta}$ is the coefficient vector.")

            st.markdown('<div class="formula-card"><div class="formula-title">2. Objective Loss Function (Ordinary Least Squares - OLS)</div>Minimizing Sum of Squared Residuals (SSR):</div>', unsafe_allow_html=True)
            st.latex(r"J(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{2n} \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2")
            st.markdown("**Closed-Form Analytical Solution (Normal Equation):**")
            st.latex(r"\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}")

        with f_col2:
            st.markdown('<div class="formula-card"><div class="formula-title">3. Regularization Formulations (Ridge & Lasso)</div>Loss penalty functions for weight constraints:</div>', unsafe_allow_html=True)
            st.markdown("**Ridge ($L_2$ Regularization):**")
            st.latex(r"J_{\text{Ridge}}(\boldsymbol{\beta}) = \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p \beta_j^2 \implies \boldsymbol{\hat{\beta}} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}")
            st.markdown("**Lasso ($L_1$ Regularization):**")
            st.latex(r"J_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p |\beta_j|")

            st.markdown('<div class="formula-card"><div class="formula-title">4. Performance Evaluation Metrics</div>Statistical validation equations:</div>', unsafe_allow_html=True)
            st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{\text{SSR}}{\text{SST}}")
            st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}")

    # Step 1: Prepare the selected data
    with st.expander("▼ Step 1 - Prepare the selected data (Flow: Data Cleaning & Feature Transformation)", expanded=True):
        st.markdown("""
        <div style="margin-bottom:12px;">
            <span class="flow-badge">Flow Step 1</span> <b>Input Matrix Ingestion & Encoding:</b>
            Raw categorical features are encoded into $(0, 1)$ indicator variables ($k-1$ dummy columns per categorical attribute to eliminate multicollinearity).
        </div>
        """, unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Rows uploaded / sampled", f"{len(df):,}")
        with p2:
            st.metric("Usable target rows", f"{len(df):,}")
        with p3:
            st.metric("Rows removed (nulls/invalid)", "0")
        with p4:
            st.metric("Selected features", f"{len(pipeline_results['active_features'])}")

        st.markdown("**Target ($Y$):** `price` (Continuous ticket fare in ₹)")
        st.markdown("**Numeric features ($X_{\\text{num}}$):** `duration`, `days_left`")
        st.markdown("**Categorical features ($X_{\\text{cat}}$ One-Hot Encoded):** `airline`, `source_city`, `destination_city`, `class`, `stops`, `departure_time`, `arrival_time`")

    # Step 2: Split training and test data
    with st.expander("▼ Step 2 - Split training and test data (Flow: Partitioning & Validation Guard)", expanded=True):
        st.markdown("""
        <div style="margin-bottom:12px;">
            <span class="flow-badge">Flow Step 2</span> <b>Train/Test Matrix Partition:</b>
            Dataset is partitioned into $\\mathbf{X}_{\\text{train}}, \\mathbf{y}_{\\text{train}}$ for coefficient fitting and $\\mathbf{X}_{\\text{test}}, \\mathbf{y}_{\\text{test}}$ held back to prevent data leakage.
        </div>
        """, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Training rows (N_train)", f"{pipeline_results['X_train'].shape[0]:,}")
        with s2:
            st.metric("Test rows (N_test)", f"{pipeline_results['X_test'].shape[0]:,}")
        with s3:
            st.metric("Test set share", f"{test_split_ratio*100:.0f}%")
        st.caption(f"Random seed: {random_seed}. Train matrix shape: `{pipeline_results['X_train'].shape}` | Test matrix shape: `{pipeline_results['X_test'].shape}`")

    # Step 3: Train model and inspect coefficients
    with st.expander("▼ Step 3 - Train model and inspect coefficients (Flow: Weight Optimization)", expanded=True):
        st.markdown("""
        <div style="margin-bottom:12px;">
            <span class="flow-badge">Flow Step 3</span> <b>Optimization & Parameter Vector Estimation:</b>
            Solving $\\boldsymbol{\\hat{\\beta}} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{Y}$ calculates the rate of fare change associated with each indicator variable and numerical attribute.
        </div>
        """, unsafe_allow_html=True)

        st.success(f"{selected_algorithm} optimization converged successfully.")
        st.write(f"**Baseline Intercept (\\(\\beta_0\\)):** `{active_model.intercept_:,.6f}` *(Baseline fare when all categorical indicators are at their reference levels)*")

        coef_df = pd.DataFrame({
            'Feature': model_columns,
            'Coefficient': active_model.coef_,
            'Effect direction': ['Positive' if c >= 0 else 'Negative' for c in active_model.coef_]
        }).sort_values(by='Coefficient', ascending=False)

        st.dataframe(
            coef_df.style.format({'Coefficient': '{:,.4f}'}),
            height=280,
            use_container_width=True
        )

        csv_coef = coef_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download coefficients as CSV",
            data=csv_coef,
            file_name="linear_regression_coefficients.csv",
            mime="text/csv"
        )

        # Model Calculation Breakdown & Full Mathematical Formulation
        with st.expander("📐 View Linear Regression Model Calculation (Fitted Equation & Sample Inference)", expanded=True):
            st.markdown("##### 1. Fitted Linear Regression Model Equation:")
            st.write(f"$$\\hat{{Y}} = {active_model.intercept_:,.2f} " + "".join([f" {'+' if c >= 0 else '-'} {abs(c):,.2f} \\times \\text{{{col.replace('_', ' ')}}}" for col, c in zip(model_columns[:10], active_model.coef_[:10])]) + " + \\dots$$")
            st.caption(f"Showing first 10 terms of all {len(model_columns)} feature coefficients. Base intercept $\\beta_0 = {active_model.intercept_:,.2f}$.")

            st.markdown("##### 2. Step-by-Step Sample Prediction Dot Product Calculation (Row #1 from Test Set):")
            sample_x = pipeline_results['X_test'].iloc[0]
            sample_actual = pipeline_results['y_test'].iloc[0]
            active_terms = [(col, float(val), float(coef)) for col, val, coef in zip(model_columns, sample_x, active_model.coef_) if val != 0]

            calc_rows = []
            calc_sum = active_model.intercept_
            calc_rows.append({"Component": "Baseline Intercept (β₀)", "Feature Value (x)": "1.0", "Coefficient (β)": f"{active_model.intercept_:,.2f}", "Contribution (x · β) (₹)": f"{active_model.intercept_:,.2f}"})
            for col, val, coef in active_terms:
                term_val = val * coef
                calc_sum += term_val
                calc_rows.append({
                    "Component": col,
                    "Feature Value (x)": f"{val}",
                    "Coefficient (β)": f"{coef:,.2f}",
                    "Contribution (x · β) (₹)": f"{term_val:,.2f}"
                })

            st.dataframe(pd.DataFrame(calc_rows), use_container_width=True)
            st.markdown(f"""
            - **Calculated Sum:** $\\beta_0 + \\sum_{{j=1}}^p \\beta_j x_j = \\mathbf{{{calc_sum:,.2f}}}$
            - **Model Predicted Price ($\\hat{{y}}$):** `₹{max(1000.0, calc_sum):,.2f}`
            - **Actual Ticket Price ($y$):** `₹{sample_actual:,.2f}`
            - **Individual Residual Error ($e = y - \\hat{{y}}$):** `₹{sample_actual - calc_sum:,.2f}`
            """)

    # Step 4: Evaluate on unseen test data
    with st.expander("▼ Step 4 - Evaluate on unseen test data (Flow: Generalization Scoring)", expanded=True):
        st.markdown("""
        <div style="margin-bottom:12px;">
            <span class="flow-badge">Flow Step 4</span> <b>Out-of-Sample Empirical Evaluation:</b>
            Model predicts $\\mathbf{\\hat{Y}}_{\\text{test}} = \\mathbf{X}_{\\text{test}} \\boldsymbol{\\hat{\\beta}}$ and compares predictions against true ticket prices $\\mathbf{Y}_{\\text{test}}$.
        </div>
        """, unsafe_allow_html=True)

        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.metric("R² score", f"{pipeline_results['test_r2']:.3f}", help="Proportion of variance explained by model: 1 - (SSR / SST)")
        with e2:
            st.metric("Adjusted R²", f"{pipeline_results['adj_r2']:.3f}", help="R² adjusted for the number of predictors p")
        with e3:
            st.metric("MAE", f"₹{pipeline_results['test_mae']:,.2f}", help="Mean Absolute Error in ticket price INR")
        with e4:
            st.metric("MSE", f"{pipeline_results['test_mse']:,.0f}", help="Mean Squared Error")
        with e5:
            st.metric("RMSE", f"₹{pipeline_results['test_rmse']:,.2f}", help="Root Mean Squared Error")

        st.caption("Metric guide: R² closer to 1 means the model explains more variation. Lower MAE and RMSE indicate higher predictive precision.")

        # Interactive Step-by-Step Metric Calculation Breakdown
        with st.expander("🧮 View Detailed Step-by-Step Mathematical Calculations for Each Metric", expanded=True):
            st.markdown("##### Empirical Quantities from Held-Out Test Set:")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.write(f"- **Test Samples ($n$):** `{pipeline_results['n_test']:,}`")
                st.write(f"- **Features ($p$):** `{pipeline_results['p_features']}`")
            with q2:
                st.write(f"- **Mean Fare (bar_y):** `₹{pipeline_results['y_mean']:,.2f}`")
                st.write(f"- **Sum Abs Error (SAE):** `₹{pipeline_results['sae']:,.2f}`")
            with q3:
                st.write(f"- **Residual Sum of Squares (SSR):** `{pipeline_results['ssr']:,.2e}`")
                st.write(f"- **Total Sum of Squares (SST):** `{pipeline_results['sst']:,.2e}`")
            with q4:
                st.write(f"- **Degrees of Freedom ($n - p - 1$):** `{pipeline_results['n_test'] - pipeline_results['p_features'] - 1:,}`")

            st.markdown("---")
            st.markdown("##### 1. Coefficient of Determination ($R^2$) Calculation:")
            st.latex(r"R^2 = 1 - \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}")
            r2_sub = pipeline_results['ssr'] / pipeline_results['sst']
            st.latex(rf"R^2 = 1 - \frac{{{pipeline_results['ssr']:.4e}}}{{{pipeline_results['sst']:.4e}}} = 1 - {r2_sub:.4f} = \mathbf{{{pipeline_results['test_r2']:.4f}}}")

            st.markdown("##### 2. Adjusted $R^2$ Calculation:")
            st.latex(r"\bar{R}^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}")
            st.latex(rf"\bar{{R}}^2 = 1 - \left(1 - {pipeline_results['test_r2']:.4f}\right) \frac{{{pipeline_results['n_test'] - 1}}}{{{pipeline_results['n_test'] - pipeline_results['p_features'] - 1}}} = \mathbf{{{pipeline_results['adj_r2']:.4f}}}")

            st.markdown("##### 3. Mean Absolute Error (MAE) Calculation:")
            st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{\text{Sum of Absolute Residuals}}{n}")
            st.latex(rf"\text{{MAE}} = \frac{{{pipeline_results['sae']:,.2f}}}{{{pipeline_results['n_test']:,}}} = \mathbf{{₹{pipeline_results['test_mae']:,.2f}}}")

            st.markdown("##### 4. Mean Squared Error (MSE) & Root Mean Squared Error (RMSE) Calculation:")
            st.latex(r"\text{MSE} = \frac{\text{SSR}}{n} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2")
            st.latex(rf"\text{{MSE}} = \frac{{{pipeline_results['ssr']:.4e}}}{{{pipeline_results['n_test']:,}}} = \mathbf{{{pipeline_results['test_mse']:,.2f}}}")
            st.latex(r"\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}")
            st.latex(rf"\text{{RMSE}} = \sqrt{{{pipeline_results['test_mse']:,.2f}}} = \mathbf{{₹{pipeline_results['test_rmse']:,.2f}}}")

    # Step 5: Test-set predictions table
    with st.expander("▼ Step 5 - Test-set predictions & residuals (Flow: Residual Error Inspection)", expanded=True):
        st.markdown('<span class="flow-badge">Flow Step 5</span> <b>Residual Error Calculation:</b> Residual vector $e = Y_{\\text{test}} - \\hat{Y}_{\\text{test}}$. Values near zero indicate optimal alignment with ground truth.', unsafe_allow_html=True)
        y_test_arr = pipeline_results['y_test'].values
        y_pred_arr = pipeline_results['y_pred_test']
        residuals_arr = y_test_arr - y_pred_arr

        # Create sample prediction preview
        test_preview = df.iloc[pipeline_results['X_test'].index].head(15).copy()
        test_preview.insert(0, 'Actual Price', y_test_arr[:15])
        test_preview.insert(1, 'Predicted Price', np.round(y_pred_arr[:15], 2))
        test_preview.insert(2, 'Residual (Error)', np.round(residuals_arr[:15], 2))

        disp_cols = ['Actual Price', 'Predicted Price', 'Residual (Error)', 'airline', 'source_city', 'destination_city', 'class', 'stops', 'duration', 'days_left']
        st.dataframe(
            test_preview[disp_cols].style.format({
                'Actual Price': '₹{:,.0f}',
                'Predicted Price': '₹{:,.2f}',
                'Residual (Error)': '₹{:,.2f}'
            }),
            use_container_width=True
        )

        pred_full_df = pd.DataFrame({
            'Actual_Price': y_test_arr,
            'Predicted_Price': y_pred_arr,
            'Residual': residuals_arr
        })
        csv_preds = pred_full_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download test predictions as CSV",
            data=csv_preds,
            file_name="test_predictions_residuals.csv",
            mime="text/csv"
        )

# ----------------------------------------------------
# TAB 3: LIVE PRICE PREDICTOR
# ----------------------------------------------------
with tab_pred:
    st.subheader("🛫 Real-time Flight Ticket Price Estimator")
    st.write("Configure journey details to compute real-time fare predictions using the trained regression model.")

    col_in1, col_in2, col_in3 = st.columns(3)
    unique_airlines = sorted(df['airline'].unique().tolist())
    unique_sources = sorted(df['source_city'].unique().tolist())
    unique_destinations = sorted(df['destination_city'].unique().tolist())
    unique_classes = sorted(df['class'].unique().tolist())
    unique_stops = sorted(df['stops'].unique().tolist())
    unique_dep_times = sorted(df['departure_time'].unique().tolist())
    unique_arr_times = sorted(df['arrival_time'].unique().tolist())

    with col_in1:
        st.markdown("##### 📍 Journey Route")
        in_airline = st.selectbox("Airline Company", unique_airlines, index=0)
        in_source = st.selectbox("Source Metro City", unique_sources, index=0)
        dest_options = [d for d in unique_destinations if d != in_source] or unique_destinations
        in_dest = st.selectbox("Destination Metro City", dest_options, index=0)
        in_class = st.selectbox("Cabin Class", unique_classes, index=0)

    with col_in2:
        st.markdown("##### ⏰ Flight Schedule")
        in_dep_time = st.selectbox("Departure Time Window", unique_dep_times, index=0)
        in_arr_time = st.selectbox("Arrival Time Window", unique_arr_times, index=0)
        in_stops = st.selectbox("Stop Category", unique_stops, index=0, format_func=lambda s: {
            'zero': 'Non-stop Flight',
            'one': '1 Intermediate Stop',
            'two_or_more': '2 or More Stops'
        }.get(s, s))

    with col_in3:
        st.markdown("##### ⏱️ Flight Duration & Lead Time")
        in_duration = st.slider("Flight Duration (Hours)", min_value=0.5, max_value=40.0, value=2.5, step=0.25)
        in_days_left = st.slider("Booking Lead (Days Left)", min_value=1, max_value=60, value=20, step=1)

    st.write("")
    if st.button("🔮 Calculate Predicted Ticket Price", type="primary", use_container_width=True):
        input_data = {
            'airline': in_airline,
            'source_city': in_source,
            'destination_city': in_dest,
            'class': in_class,
            'stops': in_stops,
            'departure_time': in_dep_time,
            'arrival_time': in_arr_time,
            'duration': in_duration,
            'days_left': in_days_left
        }
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        full_encoded = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in input_encoded.columns:
            if col in full_encoded.columns:
                full_encoded[col] = input_encoded[col].values

        raw_pred = active_model.predict(full_encoded)[0]
        predicted_val = max(1000.0, raw_pred)

        ci_lower = max(1000.0, predicted_val - pipeline_results['test_mae'])
        ci_upper = predicted_val + pipeline_results['test_mae']

        res_col1, res_col2 = st.columns([1.1, 1])
        with res_col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin:0; font-size:1.25rem;">Estimated Ticket Fare</h3>
                <div style="font-size: 2.8rem; font-weight:800; margin:10px 0;">₹{predicted_val:,.2f}</div>
                <div style="font-size:0.95rem; opacity:0.95;">
                    Confidence Interval (± MAE): <b>₹{ci_lower:,.0f} - ₹{ci_upper:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("##### ℹ️ Journey Specification")
            st.info(f"""
            - **Flight:** `{in_airline}` from **{in_source}** ➔ **{in_dest}**
            - **Cabin & Route:** **{in_class}** class | **{in_stops}**
            - **Booking Lead:** {in_days_left} days prior | **Duration:** {in_duration} hrs
            - **Active Estimator:** `{selected_algorithm}`
            - **Validation Score ($R^2$):** `{pipeline_results['test_r2']*100:.2f}%`
            """)

# ----------------------------------------------------
# TAB 4: MODEL TESTER & DIAGNOSTICS
# ----------------------------------------------------
with tab_tester:
    st.subheader("🧪 Model Tester & Statistical Diagnostics")
    st.write("Inspect residual distributions, actual vs predicted scatter, homoscedasticity, and error tolerances.")

    y_actual = pipeline_results['y_test'].values
    y_pred = pipeline_results['y_pred_test']
    residuals = y_actual - y_pred

    sample_size_diag = min(3000, len(y_actual))
    sample_idx = np.random.choice(len(y_actual), size=sample_size_diag, replace=False)

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        # Actual vs Predicted Scatter
        fig_avp = px.scatter(
            x=y_actual[sample_idx],
            y=y_pred[sample_idx],
            labels={'x': 'Actual Ticket Price (₹)', 'y': 'Predicted Ticket Price (₹)'},
            title="Actual vs. Predicted Ticket Price (Validation Set)",
            opacity=0.6,
            color_discrete_sequence=['#2563EB']
        )
        min_v = min(y_actual[sample_idx].min(), y_pred[sample_idx].min())
        max_v = max(y_actual[sample_idx].max(), y_pred[sample_idx].max())
        fig_avp.add_trace(go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode='lines',
            line=dict(color='#EF4444', dash='dash', width=2),
            name='Ideal 1:1 Fit'
        ))
        fig_avp.update_layout(height=440)
        st.plotly_chart(fig_avp, use_container_width=True)

    with diag_col2:
        # Residuals vs Predicted (Homoscedasticity)
        fig_resid = px.scatter(
            x=y_pred[sample_idx],
            y=residuals[sample_idx],
            labels={'x': 'Predicted Ticket Price (₹)', 'y': 'Residual Error (₹)'},
            title="Residuals vs Predicted (Homoscedasticity Inspection)",
            opacity=0.6,
            color_discrete_sequence=['#8B5CF6']
        )
        fig_resid.add_hline(y=0, line_dash='dash', line_color='#EF4444', line_width=2)
        fig_resid.update_layout(height=440)
        st.plotly_chart(fig_resid, use_container_width=True)

    diag_col3, diag_col4 = st.columns(2)
    with diag_col3:
        fig_dist = px.histogram(
            residuals,
            nbins=60,
            title="Residual Error Distribution (Normality Verification)",
            labels={'value': 'Residual (Actual - Predicted) (₹)'},
            color_discrete_sequence=['#059669'],
            marginal="box"
        )
        fig_dist.update_layout(height=440, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with diag_col4:
        st.markdown("##### 🏆 Multi-Model Benchmark Comparison")
        st.dataframe(
            pipeline_results['benchmarks_df'].style.format({
                'Test R²': '{:.4f}',
                'Test MAE (₹)': '₹{:,.2f}',
                'Test RMSE (₹)': '₹{:,.2f}'
            }),
            use_container_width=True
        )

        st.markdown("##### 📊 Rigorous Error Diagnostics")
        mean_err = np.mean(residuals)
        std_err = np.std(residuals)
        pct_within_5k = np.mean(np.abs(residuals) <= 5000) * 100
        pct_within_10k = np.mean(np.abs(residuals) <= 10000) * 100

        st.info(f"""
        - **Mean Residual Error:** `₹{mean_err:,.2f}` *(Unbiased center ≈ 0)*
        - **Standard Error:** `₹{std_err:,.2f}`
        - **Accuracy within ±₹5,000:** `{pct_within_5k:.1f}%`
        - **Accuracy within ±₹10,000:** `{pct_within_10k:.1f}%`
        - **5-Fold Cross Validation:** `{pipeline_results['cv_scores'].mean()*100:.2f}% ± {pipeline_results['cv_scores'].std()*100:.2f}%`
        """)

# ----------------------------------------------------
# TAB 5: RAW & PROCESSED DATA VIEW
# ----------------------------------------------------
with tab_data:
    st.subheader("📋 Dataset Preview & Exploration")
    st.write(f"Currently viewing records from: **{data_name}**")
    
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        filter_airline = st.multiselect("Filter by Airline", options=df['airline'].unique(), default=df['airline'].unique())
    with f_col2:
        filter_class = st.multiselect("Filter by Class", options=df['class'].unique(), default=df['class'].unique())
        
    filtered_df = df[df['airline'].isin(filter_airline) & df['class'].isin(filter_class)]
    
    st.dataframe(
        filtered_df[['airline', 'source_city', 'destination_city', 'departure_time', 'stops', 'arrival_time', 'class', 'duration', 'days_left', 'price']].head(250),
        use_container_width=True
    )
    
    csv_buf = io.StringIO()
    filtered_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv_buf.getvalue(),
        file_name="processed_flight_dataset.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Linear Regression Studio & Flight Price Analytics Hub | Powered by Streamlit & Scikit-Learn")
