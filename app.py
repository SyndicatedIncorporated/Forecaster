import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Enterprise Demand Planner (Relex Style)")

# --- SIDEBAR: Forecast Parameters (Like your image) ---
st.sidebar.header("Forecast Settings")
model_type = st.sidebar.selectbox("Smoothing Type", ["Additive", "Multiplicative"])
alpha = st.sidebar.slider("Alpha (Level)", 0.0, 1.0, 0.6)
beta = st.sidebar.slider("Beta (Trend)", 0.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma (Season)", 0.0, 1.0, 0.2)
forecast_ahead = st.sidebar.number_input("Forecast Ahead (Days/Months)", value=30)
outlier_threshold = st.sidebar.number_input("Outlier Threshold", value=3.5)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Sales History (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Relex-style mapping
    date_col = st.selectbox("Date Column", df.columns)
    sales_col = st.selectbox("Sales Column", df.columns)
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # --- THE ENGINE: Holt-Winters Exponential Smoothing ---
    with st.spinner('Calculating Forecast...'):
        model = ExponentialSmoothing(
            df[sales_col], 
            trend=model_type.lower(), 
            seasonal=model_type.lower(), 
            seasonal_periods=7 # Set to 12 for monthly, 7 for weekly
        )
        # Applying your specific Alpha, Beta, Gamma from the sidebar
        fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        forecast = fit.forecast(forecast_ahead)

    # --- VISUALIZATION ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[sales_col], name="Actual Sales"))
    
    # Create dates for the forecast
    last_date = df[date_col].iloc[-1]
    forecast_dates = pd.date_range(last_date, periods=forecast_ahead + 1, freq='D')[1:]
    
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Planned Forecast", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE (Like your image) ---
    st.subheader("Forecast Parameter Log")
    log_data = {
        "Forecast Model": ["Holt-Winters"],
        "Alpha": [alpha],
        "Beta": [beta],
        "Gamma": [gamma],
        "Forecast Ahead": [forecast_ahead],
        "Demand Class": ["Seasonal" if gamma > 0.1 else "Stable"]
    }
    st.table(pd.DataFrame(log_data))

# Add a product selector
product_list = df['Product_ID'].unique()
selected_product = st.selectbox("Select Product to Forecast", product_list)

# Filter data for just that product
product_df = df[df['Product_ID'] == selected_product]

# Automatically grab the Alpha/Beta/Gamma from the CSV!
default_alpha = float(product_df['Alpha'].iloc[0])
default_beta = float(product_df['Beta'].iloc[0])
default_gamma = float(product_df['Gamma'].iloc[0])

# Update sliders to use these defaults
alpha = st.sidebar.slider("Alpha (Level)", 0.0, 1.0, default_alpha)
beta = st.sidebar.slider("Beta (Trend)", 0.0, 1.0, default_beta)
gamma = st.sidebar.slider("Gamma (Season)", 0.0, 1.0, default_gamma)

# 1. Clean up column names (removes hidden spaces like " Date ")
df.columns = df.columns.str.strip()

# 2. Convert to date, but turn "bad" data into 'NaT' (Not a Time) instead of crashing
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# 3. Drop any rows that failed to convert (like empty rows at the end of the file)
df = df.dropna(subset=[date_col])

# 4. Sort by date (crucial for forecasting)
df = df.sort_values(date_col)
