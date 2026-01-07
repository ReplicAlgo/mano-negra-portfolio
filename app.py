# app.py
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Load Excel for historical data ---
@st.cache_data
def load_data():
    raw = pd.read_excel("Mano_Negra_Portolio.xlsx", header=None)
    
    # Extract instruments from row 1 (index 1), columns 48:59
    instruments = list(raw.iloc[1, 48:59])
    
    # Historical data: rows 5+ to -1 (exclude grand total)
    asset_cols = list(range(1, 12))
    historical_df = raw.iloc[5:-1, [0] + asset_cols + [31, 32]].copy()
    historical_df.columns = ['time'] + instruments + ['year', 'hour']
    
    # Clean data
    historical_df[instruments] = historical_df[instruments].fillna(0)
    for col in instruments:
        historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce').fillna(0)
    
    historical_df['year'] = pd.to_numeric(historical_df['year'], errors='coerce').ffill()
    historical_df['hour'] = pd.to_numeric(historical_df['hour'], errors='coerce').fillna(0).astype(int)
    
    return historical_df, instruments

historical_df, instruments = load_data()

# Find index of S&P500
sp500_index = instruments.index("S&P500")

# --- Branding ---
st.image("IndicadorManoNegraBrand.png", width=900)
st.markdown("<h2 style='text-align: center; margin-top:30px;'>Mano Negra Portfolio Simulation</h2>", unsafe_allow_html=True)

# --- Sidebar: Portfolio Settings ---
st.sidebar.header("Portfolio Settings")

# Trading Hours Slider - Default to full day (0-24)
trading_hours = st.sidebar.slider(
    "Select Trading Hours",
    min_value=0,
    max_value=24,
    value=(0, 24),
    help="0 to 24 means all hours are included"
)
start_hour, end_hour = trading_hours

# Allocations sliders - Default: S&P500 = 100%, others = 0%
st.sidebar.subheader("Asset Allocations (%)")
allocations = {}
for i, inst in enumerate(instruments):
    default = 100 if i == sp500_index else 0
    allocations[inst] = st.sidebar.slider(f"{inst}", 0, 200, default, step=5,
                                          help="0 = off, 100 = full size, >100 = leverage")

# Total allocation percentage
total_alloc = sum(allocations.values())
st.sidebar.markdown(f"**Total Allocation: {total_alloc}%**")

if total_alloc > 100:
    st.sidebar.warning(f"⚠️ Leverage applied: {total_alloc - 100}% over 100%")

# Compute weights (multipliers, no normalization)
weights = {inst: alloc / 100.0 for inst, alloc in allocations.items()}

# --- Dynamic title with allocation info ---
allocation_text = f"Total Allocation: {total_alloc}%"
if total_alloc > 100:
    allocation_text += f" ⚠️ (Leverage: +{total_alloc - 100}%)"

st.markdown(f"### Equity & Drawdown  —  *{allocation_text}*")

# --- Compute dynamic metrics ---
filtered_df = historical_df[
    (historical_df['hour'] >= start_hour) & 
    (historical_df['hour'] < end_hour)
].copy()

if not filtered_df.empty:
    filtered_df['weighted_pl'] = sum(filtered_df[inst] * weights[inst] for inst in instruments)
    
    cum_equity = np.cumsum(filtered_df['weighted_pl'].values)
    total_return = cum_equity[-1] if len(cum_equity) > 0 else 0.0
    
    cummax_equity = np.maximum.accumulate(cum_equity)
    drawdown = cum_equity - cummax_equity
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    
    yearly_returns = filtered_df.groupby('year')['weighted_pl'].sum().reset_index()
    yearly_df = pd.DataFrame({
        "Year": yearly_returns['year'].astype(int),
        "Return": yearly_returns['weighted_pl']
    })
    
    dates = filtered_df['time'].values
else:
    total_return = 0.0
    max_drawdown = 0.0
    cum_equity = np.array([])
    drawdown = np.array([])
    yearly_df = pd.DataFrame(columns=["Year", "Return"])
    dates = []

# --- KPIs ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style='border:2px solid #ccc; border-radius:8px; padding:10px; text-align:center;'>
            <h5>Total Return</h5>
            <p style='font-size:18px; color:green;'>${total_return:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style='border:2px solid #ccc; border-radius:8px; padding:10px; text-align:center;'>
            <h5>Max Drawdown</h5>
            <p style='font-size:18px; color:red;'>${max_drawdown:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Equity Curve & Drawdown ---
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=("Equity Curve", "Drawdown"),
    row_heights=[0.65, 0.35]
)

if len(cum_equity) > 0:
    fig.add_trace(go.Scatter(x=dates, y=cum_equity,
                             mode="lines", name="Equity",
                             line=dict(color="green")),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=drawdown,
                             fill="tozeroy", name="Drawdown",
                             opacity=0.5,
                             line=dict(color="red"),
                             fillcolor="rgba(255,0,0,0.3)"),
                  row=2, col=1)

fig.update_layout(
    height=650,
    showlegend=True,
    hovermode="x unified",
    dragmode="zoom",
    title_text=""
)

fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ]),
        bgcolor="lightgray",
        activecolor="gray"
    ),
    type="date",
    row=1, col=1
)

st.plotly_chart(fig, use_container_width=True)

# --- Yearly Returns Table with Conditional Formatting ---
st.subheader("Yearly Returns")
if not yearly_df.empty:
    # Function to color negative numbers red
    def color_negative_red(val):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'

    # Apply styling: format as currency and apply the color function
    styled_yearly_df = yearly_df.style.applymap(color_negative_red, subset=['Return'])\
                                     .format({"Return": "${:,.0f}", "Year": "{:d}"})
    
    st.table(styled_yearly_df)
else:
    st.write("No data available for the selected parameters.")
