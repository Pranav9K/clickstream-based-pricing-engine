import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# 🌟 PAGE CONFIGURATION & THEME PREFERENCE
# ----------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Engine Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Real-Time Clickstream Behavioral Pricing Engine")
st.markdown("""
This production dashboard links user clickstream metadata with machine learning to optimize item prices against consumer demand elasticity metrics.
""")

# ----------------------------------------------------
# 📥 CACHED DATA & ARTIFACT LOADING
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    # Pulling directly from your newly organized subfolder
    model = joblib.load('streamlit_assets/dynamic_pricing_rf_model.pkl')
    feature_cols = joblib.load('streamlit_assets/model_feature_schema.pkl')
    return model, feature_cols

@st.cache_data
def load_processed_data():
    # Pulling the clean CSV from your organized subfolder
    df = pd.read_csv('streamlit_assets/features_df.csv')
    return df

try:
    pricing_model, feature_cols = load_ml_assets()
    features_df = load_processed_data()
except Exception as e:
    st.error("⚠️ System Error: Missing core deployment artifacts. Ensure you ran your updated Cell 11 in your notebook!")
    st.stop()

# ----------------------------------------------------
# 🎛️ SIDEBAR CONTROL MANAGEMENT
# ----------------------------------------------------
st.sidebar.header("🎯 Item Optimization Target")

# Category Selector filter
available_categories = features_df['category'].unique()
selected_category = st.sidebar.selectbox("Filter Store Category:", available_categories)

# Filter product dataframe down to specific selection
filtered_products = features_df[features_df['category'] == selected_category]
product_choices = filtered_products['product_id'].tolist()

selected_product_id = st.sidebar.selectbox("Select Target Product ID:", product_choices)

# Extract core data metrics for the target selection from the Pandas Series
prod_data = features_df[features_df['product_id'] == selected_product_id].iloc[0]
wholesale_cost = prod_data['cost_usd']
original_price = prod_data['price_usd']
click_velocity = prod_data['click_velocity']
cart_ratio = prod_data['cart_to_view_ratio']

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Live Market Shock Simulation")
# Add interactive controls to let users simulate traffic surges in real time
velocity_multiplier = st.sidebar.slider("Simulate View Traffic Multiplier:", 0.5, 3.0, 1.0, step=0.1)
adjusted_velocity = click_velocity * velocity_multiplier

# ----------------------------------------------------
# ⚙️ ENGINE CORE CALCULATION LOOP
# ----------------------------------------------------
simulated_prices = np.linspace(wholesale_cost + 1.0, original_price * 2.5, 100)
best_price = original_price
max_expected_profit = -float('inf')
optimization_records = []

for sim_price in simulated_prices:
    sim_margin = (sim_price - wholesale_cost) / sim_price
    
    input_payload = pd.DataFrame([{
        'price_usd': sim_price,
        'cost_usd': wholesale_cost,
        'click_velocity': adjusted_velocity,
        'cart_to_view_ratio': cart_ratio,
        'gross_margin_pct': sim_margin
    }])
    
    # Predict the checkout likelihood using our Random Forest asset
    predicted_conversion = pricing_model.predict(input_payload)[0]
    expected_profit = (sim_price - wholesale_cost) * (predicted_conversion * adjusted_velocity)
    
    optimization_records.append((sim_price, expected_profit))
    
    if expected_profit > max_expected_profit:
        max_expected_profit = expected_profit
        best_price = sim_price

# ----------------------------------------------------
# 📊 UI DASHBOARD VISUAL GRID
# ----------------------------------------------------
# Row 1: High-level KPI summary scorecard widgets
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Wholesale Unit Floor Cost", value=f"${wholesale_cost:.2f}")
with col2:
    st.metric(label="Original Catalog Price", value=f"${original_price:.2f}")
with col3:
    st.metric(label="Engine Recommended Price", value=f"${best_price:.2f}", delta=f"${best_price - original_price:.2f}")
with col4:
    st.metric(label="Max Projected Profit Stream", value=f"${max_expected_profit:.2f}")

st.markdown("---")

# Row 2: Graph layouts split down the middle
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.subheader("📊 Revenue Optimization Target Wave")
    plot_df = pd.DataFrame(optimization_records, columns=['Simulated_Price', 'Expected_Profit'])
    
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(plot_df['Simulated_Price'], plot_df['Expected_Profit'], color='darkorange', linewidth=3)
    ax.axvline(x=best_price, color='red', linestyle='--', label=f'Optimal Target (${best_price:.2f})')
    ax.axvline(x=original_price, color='blue', linestyle=':', label=f'Original Catalog (${original_price:.2f})')
    ax.set_xlabel("Simulated Retail Price ($)")
    ax.set_ylabel("Projected Profit Stream ($)")
    ax.legend(fontsize=8)
    sns.despine()
    st.pyplot(fig)

with graph_col2:
    st.subheader("🎯 Consumer Engagement Metrics")
    st.write(f"**Current Catalog Segment:** `{selected_category}`")
    st.write(f"**Baseline Click Velocity:** `{click_velocity:,.0f} historical page views`")
    st.write(f"**Simulated Click Velocity Surge:** `{adjusted_velocity:,.0f} active tracking views`")
    st.write(f"**Purchase Intent Ratio (Cart to View Ratio):** `{cart_ratio * 100:.2f}%`")
    
    # Interactive optimization summary box
    if best_price > original_price:
        st.success("💡 Recommendation Summary: High demand momentum suggests consumer price tolerance can handle an increased margin premium.")
    else:
        st.warning("💡 Recommendation Summary: High demand resistance detected. Maintain closer affinity to catalog baseline to defend conversion volumes.")