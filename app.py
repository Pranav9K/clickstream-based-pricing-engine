import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# browser window configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Engine Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Real-Time Clickstream Behavioral Pricing Engine")
st.markdown("""
This production dashboard links user clickstream metadata with machine learning to optimize item prices against consumer demand elasticity metrics.
**Use the sidebar to manually input product metrics and simulate live market shocks.**
""")

# ----------------------------------------------------
# loading machine learning artifacts
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load('streamlit_assets/dynamic_pricing_rf_model.pkl')
    feature_cols = joblib.load('streamlit_assets/model_feature_schema.pkl')
    return model, feature_cols

# ----------------------------------------------------
# failsafe data import check
# ----------------------------------------------------
try:
    pricing_model, feature_cols = load_ml_assets()
except Exception as e:
    st.error("⚠️ System Error: Missing core deployment artifacts. Ensure your model is saved in the 'streamlit_assets' folder!")
    st.stop()

# ----------------------------------------------------
# sidebar navigation and manual input management
# ----------------------------------------------------
st.sidebar.header("⚙️ Manual Product Inputs")
st.sidebar.markdown("Define the baseline metrics for the product you want to analyze.")

wholesale_cost = st.sidebar.number_input("Wholesale Unit Cost ($):", min_value=0.01, value=50.00, step=1.00)
original_price = st.sidebar.number_input("Current/Original Price ($):", min_value=0.01, value=120.00, step=1.00)
click_velocity = st.sidebar.number_input("Historical Click Velocity (Page Views):", min_value=1, value=450, step=10)

# Input ratio as a percentage for better UX, then convert to decimal for the model
cart_ratio_pct = st.sidebar.slider("Purchase Intent (Cart-to-View %):", min_value=0.0, max_value=100.0, value=9.5, step=0.1)
cart_ratio = cart_ratio_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Live Market Shock Simulation")
velocity_multiplier = st.sidebar.slider("Simulate View Traffic Multiplier:", 0.5, 3.0, 1.0, step=0.1)

# calculate the new simulated click speed
adjusted_velocity = click_velocity * velocity_multiplier

# ----------------------------------------------------
# simulation and optimization engine core
# ----------------------------------------------------
# Search space: From slightly above wholesale cost up to 2.5x the original price
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
    
    # Predict conversion probability
    predicted_conversion = pricing_model.predict(input_payload)[0]
    
    # Projected profit stream
    expected_profit = (sim_price - wholesale_cost) * (predicted_conversion * adjusted_velocity)
    
    optimization_records.append((sim_price, expected_profit))
    
    if expected_profit > max_expected_profit:
        max_expected_profit = expected_profit
        best_price = sim_price

# ----------------------------------------------------
# metric card visual grid (row 1)
# ----------------------------------------------------
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

# ----------------------------------------------------
# visual charts and live performance insights (row 2)
# ----------------------------------------------------
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
    st.subheader("🎯 Custom Scenario Insights")
    st.write(f"**Current Input Margin:** `{(original_price - wholesale_cost) / original_price * 100:.2f}%`")
    st.write(f"**Baseline Click Velocity:** `{click_velocity:,.0f} historical page views`")
    st.write(f"**Simulated Click Velocity Surge:** `{adjusted_velocity:,.0f} active tracking views`")
    st.write(f"**Purchase Intent Ratio:** `{cart_ratio * 100:.2f}%`")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if best_price > original_price:
        st.success("💡 **Recommendation Summary:** The engine suggests raising the price. Projected demand elasticity at these specific input levels can sustain a higher margin without destroying total volume.")
    elif best_price < original_price:
        st.warning("💡 **Recommendation Summary:** High demand resistance detected. Lowering the price to the recommended target will likely increase volume enough to net a higher total profit.")
    else:
        st.info("💡 **Recommendation Summary:** Current pricing is perfectly optimized for maximum profit generation given the inputted conversion metrics.")