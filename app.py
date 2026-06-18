import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# Browser Window Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Seller Pricing Optimizer",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI Pricing Optimizer")
st.markdown("""
Welcome! This tool helps you find the perfect selling price for your products. 
Use the sidebar on the left to input your product's details, and the AI will calculate the price that maximizes your total profit.
""")

# ----------------------------------------------------
# Loading Machine Learning Artifacts
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load('streamlit_assets/dynamic_pricing_rf_model.pkl')
    feature_cols = joblib.load('streamlit_assets/model_feature_schema.pkl')
    return model, feature_cols

try:
    pricing_model, feature_cols = load_ml_assets()
except Exception as e:
    st.error("⚠️ System Error: Missing AI model files. Ensure your model is saved in the 'streamlit_assets' folder!")
    st.stop()

# ----------------------------------------------------
# Sidebar: Step-by-Step Seller Inputs
# ----------------------------------------------------
st.sidebar.header("Step 1: Product Details")

wholesale_cost = st.sidebar.number_input(
    "1. Unit Cost ($):", 
    min_value=0.01, value=50.00, step=1.00,
    help="How much does it cost you to manufacture or buy one unit of this product?"
)

original_price = st.sidebar.number_input(
    "2. Current Selling Price ($):", 
    min_value=0.01, value=120.00, step=1.00,
    help="What price are you currently selling this product for?"
)

st.sidebar.markdown("---")
st.sidebar.header("Step 2: Store Traffic")

click_velocity = st.sidebar.number_input(
    "3. Daily Page Views:", 
    min_value=1, value=450, step=10,
    help="On average, how many times is this product's page viewed per day?"
)

cart_ratio_pct = st.sidebar.slider(
    "4. Add-to-Cart Rate (%):", 
    min_value=0.0, max_value=100.0, value=9.5, step=0.1,
    help="Out of 100 people who view the product, how many add it to their cart?"
)
cart_ratio = cart_ratio_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.header("Step 3: Market Scenarios")
with st.sidebar.expander("Test Traffic Surges/Drops"):
    velocity_multiplier = st.slider(
        "Traffic Multiplier:", 
        0.5, 3.0, 1.0, step=0.1,
        help="Simulate what happens if a marketing campaign suddenly doubles your traffic (2.0x) or if traffic drops in half (0.5x)."
    )

adjusted_velocity = click_velocity * velocity_multiplier

# ----------------------------------------------------
# Engine Core: Calculating the Best Price
# ----------------------------------------------------
simulated_prices = np.linspace(wholesale_cost + 1.0, original_price * 2.5, 100)
best_price = original_price
max_expected_profit = -float('inf')
optimization_records = []

# Calculate current baseline metrics to compare against
current_margin = (original_price - wholesale_cost) / original_price
current_payload = pd.DataFrame([{
    'price_usd': original_price, 'cost_usd': wholesale_cost,
    'click_velocity': adjusted_velocity, 'cart_to_view_ratio': cart_ratio,
    'gross_margin_pct': current_margin
}])
current_conversion = pricing_model.predict(current_payload)[0]
current_profit = (original_price - wholesale_cost) * (current_conversion * adjusted_velocity)

# Loop to find the optimal price
for sim_price in simulated_prices:
    sim_margin = (sim_price - wholesale_cost) / sim_price
    
    input_payload = pd.DataFrame([{
        'price_usd': sim_price,
        'cost_usd': wholesale_cost,
        'click_velocity': adjusted_velocity,
        'cart_to_view_ratio': cart_ratio,
        'gross_margin_pct': sim_margin
    }])
    
    predicted_conversion = pricing_model.predict(input_payload)[0]
    expected_profit = (sim_price - wholesale_cost) * (predicted_conversion * adjusted_velocity)
    
    optimization_records.append((sim_price, expected_profit))
    
    if expected_profit > max_expected_profit:
        max_expected_profit = expected_profit
        best_price = sim_price

# ----------------------------------------------------
# Main Dashboard UI
# ----------------------------------------------------

# Top Summary Alert
profit_lift = max_expected_profit - current_profit

if profit_lift > 0:
    st.success(f"🎉 **Action Recommended:** Changing your price to **${best_price:.2f}** could generate an estimated **${profit_lift:,.2f}** in additional profit per cycle!")
else:
    st.info(f"✅ **Action Recommended:** Keep your price at **${original_price:.2f}**. You are currently priced perfectly for maximum profit.")

st.markdown("### 📊 Strategy Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏢 Current Strategy")
    st.metric(label="Current Selling Price", value=f"${original_price:.2f}")
    st.metric(label="Projected Total Profit", value=f"${current_profit:,.2f}")

with col2:
    st.markdown("#### 🚀 Recommended Strategy")
    st.metric(label="Optimized Target Price", value=f"${best_price:.2f}", delta=f"${best_price - original_price:.2f}")
    st.metric(label="Max Projected Profit", value=f"${max_expected_profit:,.2f}", delta=f"${profit_lift:,.2f} Lift")

st.markdown("---")

# ----------------------------------------------------
# Visual Charting
# ----------------------------------------------------
st.subheader("📈 Demand vs. Profit Curve")
st.markdown("This chart visualizes how high you can raise your price before buyer resistance causes your total profits to drop.")

plot_df = pd.DataFrame(optimization_records, columns=['Simulated_Price', 'Expected_Profit'])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plot_df['Simulated_Price'], plot_df['Expected_Profit'], color='#2ca02c', linewidth=3)

# Highlight zones
ax.axvline(x=best_price, color='#d62728', linestyle='--', label=f'Optimal Price (${best_price:.2f})')
ax.axvline(x=original_price, color='#1f77b4', linestyle=':', label=f'Current Price (${original_price:.2f})')

# Clean up chart aesthetics
ax.set_xlabel("Potential Retail Price ($)", fontsize=10, fontweight='bold')
ax.set_ylabel("Estimated Profit ($)", fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(fontsize=10)
sns.despine() 

st.pyplot(fig)