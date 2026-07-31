import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Climate Forecasting Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- CHECK FOR REQUIRED FILES ---
REGISTRY_FILE = "model_registry_summary.csv"
DATA_FILE = "Final_Cleaned_Environment_Temperature_Data.csv"
MODEL_DIR = "pickled_models"

if not os.path.exists(REGISTRY_FILE) or not os.path.exists(DATA_FILE):
    st.error(f"❌ Missing required files! Make sure `{REGISTRY_FILE}` and `{DATA_FILE}` are pushed to your GitHub repository.")
    st.stop()

# --- LOAD DATA & REGISTRY ---
@st.cache_data
def load_metadata():
    registry = pd.read_csv(REGISTRY_FILE)
    raw_data = pd.read_csv(DATA_FILE)
    return registry, raw_data

registry_df, historical_df = load_metadata()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Model Controls")

# 1. Select Area (Country/Region)
unique_areas = sorted(registry_df['Area'].unique())
selected_area = st.sidebar.selectbox("Select Region / Country", unique_areas)

# 2. Filter Categories available for that specific area
available_categories = sorted(registry_df[registry_df['Area'] == selected_area]['Category'].unique())
selected_category = st.sidebar.selectbox("Select Temporal Category", available_categories)

# 3. Forecast Horizon Slider
forecast_years = st.sidebar.slider("Future Forecast Horizon (Years)", min_value=5, max_value=30, value=15, step=5)

# --- FETCH MODEL METADATA ---
model_meta = registry_df[
    (registry_df['Area'] == selected_area) & 
    (registry_df['Category'] == selected_category)
].iloc[0]

# --- MAIN DASHBOARD LAYOUT ---
st.title(f"🌍 Climate Trend Analysis: {selected_area}")
st.markdown(f"**Temporal Slice:** `{selected_category}` | **Deployment Model:** Meta Prophet")

# Display Performance Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Validation MAE", f"{model_meta['Validation_MAE']:.4f} °C")
col2.metric("Validation RMSE", f"{model_meta['Validation_RMSE']:.4f} °C")
col3.metric("Optimal CPS (Flexibility)", f"{model_meta['Best_CPS']}")
col4.metric("Optimal CPR (Range)", f"{model_meta['Best_CPR']}")

st.markdown("---")

# --- DYNAMICALLY REBUILD PATH FOR LINUX/GITHUB ---
# This ignores the Windows path in the CSV and builds an OS-safe path
safe_area = str(selected_area).replace(' ', '_').replace('/', '_')
safe_cat = str(selected_category).replace(' ', '_')
pkl_filename = f"Prophet_{safe_area}_{safe_cat}.pkl"
model_path = os.path.join(MODEL_DIR, pkl_filename)

# --- LOAD PICKLED MODEL & PREDICT ---
@st.cache_resource
def load_pickled_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if os.path.exists(model_path):
    model = load_pickled_model(model_path)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_years, freq='YS')
    forecast = model.predict(future)
    
    # Isolate actual historical data for plotting comparison
    hist_subset = historical_df[
        (historical_df['Area'] == selected_area) & 
        (historical_df['Months'] == selected_category)
    ].sort_values('Year')
    
    # --- PLOTLY INTERACTIVE CHART ---
    fig = go.Figure()

    # 1. Uncertainty Interval (Confidence Band)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='tozerox',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Uncertainty Interval (80%)'
    ))

    # 2. Historical Actual Data Points
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(hist_subset['Year'].astype(str) + '-01-01'),
        y=hist_subset['Temperature_Change'],
        mode='markers+lines',
        name='Historical Actuals (1961–2019)',
        line=dict(color='black', width=2),
        marker=dict(size=6)
    ))

    # 3. Model Fit / Future Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Trend & Forecast',
        line=dict(color='#0066cc', width=3)
    ))

    fig.update_layout(
        title=f"Temperature Anomaly Trajectory for {selected_area} ({selected_category})",
        xaxis_title="Year",
        yaxis_title="Temperature Change (°C relative to baseline)",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE VIEW ---
    with st.expander("📊 View Raw Forecast Data Table"):
        display_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        display_df['Year'] = display_df['ds'].dt.year
        display_df = display_df[display_df['Year'] >= 2020][['Year', 'yhat', 'yhat_lower', 'yhat_upper']]
        display_df.columns = ['Year', 'Predicted Temperature Change (°C)', 'Lower Bound', 'Upper Bound']
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

else:
    st.error(f"Pickled model file not found at path: `{model_path}`. Ensure your `pickled_models` folder is successfully pushed to GitHub.")