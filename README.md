# 🚀 Automated Clickstream Behavioral Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-lightgrey.svg)](https://pandas.pydata.org/)

## 📖 Project Overview
This repository contains an enterprise-grade data pipeline and optimization system that utilizes machine learning to reverse-engineer **Consumer Demand Elasticity**. By integrating over 760,000 raw, sequential user clickstream rows with catalog attributes, the system tracks fine-grained behavior metrics—such as real-time page traffic momentum and cart abandonment ratios—to dynamically predict item conversion rates and maximize overall expected store profit.

Rather than relying on static or rule-based markups, the backend trains a **Random Forest Regressor** to predict checkout likelihood across 100 simulated pricing variations per product. An automated decision loop then isolates the exact retail pricing "sweet spot" while enforcing a strict, defensive wholesale cost floor buffer to ensure risk-free margin protection.

The entire architecture is wrapped inside an interactive **Streamlit web application**, allowing product managers to monitor behavioral KPIs and simulate real-time market traffic shocks.

---

## 🎯 Architecture & Data Engineering Concept

A critical challenge in production e-commerce logging pipelines is that completed sales transactions are recorded at the **session level**. The final purchase confirmation row contains the total order checkout value, but individual item identifiers are left empty.

To connect final conversions back to specific products, this pipeline implements a **Session-Lookback Attribution Mapping Strategy**:
1. It aggregates and extracts the unique set of all `session_id` tokens that successfully finalized a purchase event.
2. It looks back at item-level events within those sessions to identify which specific products were placed into a cart (`add_to_cart`).
3. It attributes the purchase credit to those products, reducing massive raw event feeds into static, clean behavioral features per catalog ID.

### Engineered Input Model Features ($X$):
* **`price_usd`**: Current simulated retail value evaluated by the loop.
* **`cost_usd`**: Fixed wholesale cost floor defining break-even boundaries.
* **`click_velocity`**: Total historical user page view volume (Browsing Momentum).
* **`cart_to_view_ratio`**: Percentage of page views that progress to a cart addition (Purchase Intent Indicator).
* **`gross_margin_pct`**: The simulated product profit cushion evaluated as:  
  $$\text{Margin} = \frac{\text{Price} - \text{Cost}}{\text{Price}}$$

### Target Variable ($y$):
* **`conversion_rate`**: The purchase likelihood probability (ranging from `0.0` to `1.0`) given a product's state and active pricing.

---

## 📂 Repository Directory Structure

The repository maintains a clean, decoupled layout keeping your operational code, web dashboard components, and serialized data assets isolated:

```text
clickstream-pricing-engine/
│
├── clickstream_pricing_engine.ipynb  # Jupyter Notebook pipeline (ETL, Training, & Audit)
├── app.py                            # Operational interactive Streamlit web dashboard
│
├── Visualizations/                   # High-Definition system charts and evaluation graphs
│   ├── price_distribution_chart.png
│   └── revenue_optimization_curve.png
│
└── streamlit_assets/                 # Serialized model dependencies and data binaries
    ├── dynamic_pricing_rf_model.pkl  # Trained Random Forest Regressor binary
    ├── model_feature_schema.pkl      # Structural column sequence checklist
    └── features_df.csv               # Pre-processed tabular product data summary