#!/usr/bin/env python3
"""
app.py
------
Streamlit web application for the Dynamic Pricing Model.

Launch with:
    streamlit run app.py
"""

import sys
import os

import streamlit as st

# Make sure src/ is on the path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import find_optimal_price  # noqa: E402

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (minimal, clean) ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title  { font-size: 2.4rem; font-weight: 800; color: #1a73e8; }
    .subtitle    { font-size: 1.05rem; color: #555; margin-bottom: 1.5rem; }
    .footer-text { font-size: 0.8rem; color: #888; text-align: center; margin-top: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">💰 Dynamic Pricing Model for E-commerce</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">'
    "Enter your product details below and let the ML model recommend the "
    "<strong>optimal price</strong> that maximises predicted revenue."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Product Details")

col1, col2 = st.columns(2)

with col1:
    demand = st.number_input(
        "📦 Demand Level",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Current demand score for this product (10 – 500).",
    )
    season = st.selectbox(
        "🌤️ Season",
        options=["summer", "winter", "spring", "fall"],
        index=0,
        help="Season in which you want to price the product.",
    )

with col2:
    competitor_price = st.number_input(
        "🏷️ Competitor Price (₹)",
        min_value=100,
        max_value=1000,
        value=450,
        step=10,
        help="Your main competitor's current listing price.",
    )
    discount = st.number_input(
        "🎁 Discount (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Discount percentage you plan to offer (0 – 50 %).",
    )

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Find Optimal Price", type="primary", use_container_width=True):

    with st.spinner("Running ML model …"):
        try:
            result = find_optimal_price(
                demand=demand,
                season=season,
                competitor_price=competitor_price,
                discount=discount,
            )
        except FileNotFoundError:
            st.error(
                "❌ **Model not found.**  "
                "Please run `python mock_data.py` then `python src/train.py` first."
            )
            st.stop()
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("📊 Pricing Recommendation")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label="🏷️ Optimal Price",
            value=f"₹{result['optimal_price']:.0f}",
            delta="Best price",
        )
    with m2:
        st.metric(
            label="📦 Predicted Sales",
            value=f"{result['predicted_sales']} units",
            delta="At optimal price",
        )
    with m3:
        st.metric(
            label="Predicted Revenue",
            value=f"Rs. {result['predicted_revenue']:,.0f}",
            delta=f"+{result['increase_percent']}%",
        )

    # Success message
    st.success(
        f"✅ **Recommendation:** Set your price to **₹{result['optimal_price']:.0f}** "
        f"to achieve an estimated **{result['predicted_sales']} units** sold — "
        f"about **Rs. {result['predicted_revenue']:,.0f}** revenue, "
        f"a **{result['increase_percent']}%** revenue improvement over mid-range pricing."
    )

    # Explainability section
    st.subheader("🤔 Why This Price?")
    st.info(
        f"""
**How the model chose ₹{result['optimal_price']:.0f}:**

- 📦 **Demand** ({demand}) signals strong buyer interest — the model can recommend a competitive price without sacrificing volume.
- 🏷️ **Competitor price** (₹{competitor_price}) sets the market ceiling. Pricing below it captures price-sensitive shoppers.
- 🎁 **Discount** ({discount}%) lowers the effective cost, boosting conversion rates.
- 🌤️ **Season** (*{season}*) shifts typical demand patterns, influencing the optimal price point.

The Random Forest model evaluated **{int((999 - 199) / 10 + 1)} candidate prices** and selected the one with the highest predicted revenue.
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div class="footer-text">
        💰 <strong>Dynamic Pricing Model for E-commerce</strong> &nbsp;|&nbsp;
        Built with Python · Random Forest · Streamlit &nbsp;|&nbsp;
        <a href="https://github.com/yourusername/dynamic-pricing-model" target="_blank">GitHub</a>
        &nbsp;|&nbsp;
        <a href="#" target="_blank">Live Demo</a>
    </div>
    """,
    unsafe_allow_html=True,
)
