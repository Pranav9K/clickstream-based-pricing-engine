import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# browser window configuration
# ----------------------------------------------------
# this sets up the webpage title, the browser tab icon, and forces the 
# application to use a wide-screen layout instead of a narrow, centered column.
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
# loading machine learning artifacts
# ----------------------------------------------------
# we use st.cache_resource here so that the trained random forest model and 
# its specific feature schema are only loaded into system memory once. this prevents 
# the app from slowing down every single time a user clicks a button.
@st.cache_resource
def load_ml_assets():
    # pulling the trained random forest model file and the saved list of 
    # expected model feature columns directly from your streamlit_assets subfolder.
    model = joblib.load('streamlit_assets/dynamic_pricing_rf_model.pkl')
    feature_cols = joblib.load('streamlit_assets/model_feature_schema.pkl')
    return model, feature_cols

# ----------------------------------------------------
# loading historical consumer data
# ----------------------------------------------------
# we use st.cache_data to keep the cleaned dataframe stored in cache. this 
# ensures we do not spend time re-reading the csv file during ui re-renders.
@st.cache_data
def load_processed_data():
    # reading the structured csv file containing historical product metrics 
    # and clickstream data from the dedicated subfolder.
    df = pd.read_csv('streamlit_assets/features_df.csv')
    return df

# ----------------------------------------------------
# failsafe data import check
# ----------------------------------------------------
# this block attempts to load the machine learning model and the dataset. if 
# any files are missing, it catches the error safely, displays a clear 
# troubleshooting message to the end user, and halts execution immediately.
try:
    pricing_model, feature_cols = load_ml_assets()
    features_df = load_processed_data()
except Exception as e:
    st.error("⚠️ System Error: Missing core deployment artifacts. Ensure you ran your updated Cell 11 in your notebook!")
    st.stop()

# ----------------------------------------------------
# sidebar navigation and input management
# ----------------------------------------------------
# this section builds the interactive controls in the left-hand sidebar, 
# allowing users to isolate individual products and simulate traffic spikes.
st.sidebar.header("🎯 Item Optimization Target")

# get every unique product category available in the dataset so we 
# can populate the first dropdown menu for easier browsing.
available_categories = features_df['category'].unique()
selected_category = st.sidebar.selectbox("Filter Store Category:", available_categories)

# narrow down the master dataframe so it only contains items that 
# belong to the particular store category selected by the user above.
filtered_products = features_df[features_df['category'] == selected_category]
product_choices = filtered_products['product_id'].tolist()

# display a second dropdown that lets the user select a specific item id 
# that they want to run the pricing optimization engine against.
selected_product_id = st.sidebar.selectbox("Select Target Product ID:", product_choices)

# locate the exact row of data specific to the chosen product so we can 
# extract its historical cost, selling price, and user engagement statistics.
prod_data = features_df[features_df['product_id'] == selected_product_id].iloc[0]
wholesale_cost = prod_data['cost_usd']
original_price = prod_data['price_usd']
click_velocity = prod_data['click_velocity']
cart_ratio = prod_data['cart_to_view_ratio']

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Live Market Shock Simulation")

# create a slider that lets the user simulate real-time spikes or drops in 
# website traffic, allowing them to see how high surges affect suggested prices.
velocity_multiplier = st.sidebar.slider("Simulate View Traffic Multiplier:", 0.5, 3.0, 1.0, step=0.1)

# calculate the new simulated click speed by multiplying the baseline 
# historical speed by the slider value selected by the user.
adjusted_velocity = click_velocity * velocity_multiplier

# ----------------------------------------------------
# simulation and optimization engine core
# ----------------------------------------------------
# we generate a sequence of 100 possible retail selling prices, starting 
# slightly above the wholesale floor cost and extending up to 2.5 times the 
# current catalog price. this gives us a broad search space to explore.
simulated_prices = np.linspace(wholesale_cost + 1.0, original_price * 2.5, 100)
best_price = original_price
max_expected_profit = -float('inf')
optimization_records = []

# we now loop through each of the 100 possible simulated selling prices to 
# determine which spot strikes the perfect balance between high margins and conversion.
for sim_price in simulated_prices:
    # calculate the percentage profit margin that this specific simulated 
    # retail price would generate relative to its wholesale cost floor.
    sim_margin = (sim_price - wholesale_cost) / sim_price
    
    # assemble a single-row pandas dataframe that matches the exact format 
    # and feature structural layout expected by our machine learning model.
    input_payload = pd.DataFrame([{
        'price_usd': sim_price,
        'cost_usd': wholesale_cost,
        'click_velocity': adjusted_velocity,
        'cart_to_view_ratio': cart_ratio,
        'gross_margin_pct': sim_margin
    }])
    
    # pass the simulated data into the trained random forest model to predict 
    # the probability (0% to 100%) that a customer will actually buy at this specific price.
    predicted_conversion = pricing_model.predict(input_payload)[0]
    
    # calculate the projected revenue stream by multiplying the dollar profit per unit 
    # by the total expected sales (predicted conversion rate multiplied by view velocity).
    expected_profit = (sim_price - wholesale_cost) * (predicted_conversion * adjusted_velocity)
    
    # save the price-profit pair to a list so we can plot a smooth, 
    # continuous demand optimization curve on the graphs section later.
    optimization_records.append((sim_price, expected_profit))
    
    # if this specific simulated price yields a higher profit than any previous 
    # price tested, update our trackers to lock in this spot as the new optimal value.
    if expected_profit > max_expected_profit:
        max_expected_profit = expected_profit
        best_price = sim_price

# ----------------------------------------------------
# metric card visual grid (row 1)
# ----------------------------------------------------
# divide the top row of the screen into four equal, side-by-side columns 
# to display key financial data and summary metrics scorecards clearly.
col1, col2, col3, col4 = st.columns(4)

with col1:
    # displays the absolute minimum dollar cost price allowed before selling 
    # at a loss, acting as a strict base price barrier.
    st.metric(label="Wholesale Unit Floor Cost", value=f"${wholesale_cost:.2f}")
with col2:
    # displays the standard list or catalog selling price for the product 
    # as stored in the default historical csv tables.
    st.metric(label="Original Catalog Price", value=f"${original_price:.2f}")
with col3:
    # displays the algorithm's newly recommended price alongside an automatic delta badge 
    # showing the precise directional change upwards or downwards from standard catalog pricing.
    st.metric(label="Engine Recommended Price", value=f"${best_price:.2f}", delta=f"${best_price - original_price:.2f}")
with col4:
    # displays the absolute maximum theoretical total profit stream achievable 
    # at the selected optimization peak calculated across simulated conditions.
    st.metric(label="Max Projected Profit Stream", value=f"${max_expected_profit:.2f}")

st.markdown("---")

# ----------------------------------------------------
# visual charts and live performance insights (row 2)
# ----------------------------------------------------
# split the lower portion of the screen down the middle into two matching halves 
# to support an analysis graph on the left and data readings on the right.
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.subheader("📊 Revenue Optimization Target Wave")
    # convert the cache entries saved inside our optimization loop into a structured 
    # pandas dataframe layout so it can be mapped easily to standard graphing structures.
    plot_df = pd.DataFrame(optimization_records, columns=['Simulated_Price', 'Expected_Profit'])
    
    # initialize a matplotlib figure with explicit width-to-height dimensions to ensure 
    # the line graph stays visually sharp and clean within the dashboard frame.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    
    # plot a smooth, dark-orange line that illustrates how project revenue stream peaks 
    # and then drops off rapidly should unit prices scale up past customer tolerances.
    ax.plot(plot_df['Simulated_Price'], plot_df['Expected_Profit'], color='darkorange', linewidth=3)
    
    # add a dashed red vertical boundary line showing the point along the x-axis 
    # where expected profits peak out max values.
    ax.axvline(x=best_price, color='red', linestyle='--', label=f'Optimal Target (${best_price:.2f})')
    
    # add a dotted blue vertical marker representing the standard base catalog price 
    # to allow easy comparison between optimized strategies and status quo baselines.
    ax.axvline(x=original_price, color='blue', linestyle=':', label=f'Original Catalog (${original_price:.2f})')
    
    ax.set_xlabel("Simulated Retail Price ($)")
    ax.set_ylabel("Projected Profit Stream ($)")
    ax.legend(fontsize=8)
    sns.despine() # removes the top and right borders from the plot box
    st.pyplot(fig)

with graph_col2:
    st.subheader("🎯 Consumer Engagement Metrics")
    # display the metadata metrics associated with our chosen entry point, 
    # formatting variables with commas and clean rounding indicators.
    st.write(f"**Current Catalog Segment:** `{selected_category}`")
    st.write(f"**Baseline Click Velocity:** `{click_velocity:,.0f} historical page views`")
    st.write(f"**Simulated Click Velocity Surge:** `{adjusted_velocity:,.0f} active tracking views`")
    st.write(f"**Purchase Intent Ratio (Cart to View Ratio):** `{cart_ratio * 100:.2f}%`")
    
    # ----------------------------------------------------
    # system recommendation logic engine
    # ----------------------------------------------------
    # evaluate if the optimization loop determined whether the target baseline 
    # price should scale up or stay conservative based on ongoing market shock inputs.
    if best_price > original_price:
        # if simulated values point to an optimal price higher than current standards, 
        # show a green success box indicating that high demand justifies testing higher margins.
        st.success("💡 Recommendation Summary: High demand momentum suggests consumer price tolerance can handle an increased margin premium.")
    else:
        # if conversions drop off too drastically when shifting price upwards, 
        # show a yellow warning box triggering a defensive pricing hold to protect total volume.
        st.warning("💡 Recommendation Summary: High demand resistance detected. Maintain closer affinity to catalog baseline to defend conversion volumes.")