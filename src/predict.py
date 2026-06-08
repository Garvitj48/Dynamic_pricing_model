#!/usr/bin/env python3
"""
src/predict.py
--------------
Prediction module for the Dynamic Pricing Model.

Loads the trained Random Forest model and exposes find_optimal_price(),
which sweeps candidate prices and returns the one that maximises predicted
sales volume.

Can also be run directly for a quick sanity-check:
    python src/predict.py
"""

import os
import sys
import numpy as np
import joblib

# ── Paths (relative to project root) ─────────────────────────────────────────
MODEL_PATH      = os.path.join("models", "pricing_model.pkl")
FEAT_NAMES_PATH = os.path.join("models", "feature_names.pkl")

# Price range to sweep when searching for the optimal price
PRICE_MIN  = 199
PRICE_MAX  = 999
PRICE_STEP = 10

# Season string → numeric mapping (must match mock_data.py / train.py)
SEASON_MAP = {"summer": 0, "winter": 1, "spring": 2, "fall": 3}


def load_model():
    """
    Load the trained model and feature names from disk.

    Returns
    -------
    model        : fitted RandomForestRegressor
    feature_names: list[str]

    Raises
    ------
    FileNotFoundError if model artefacts are missing.
    """
    for path in (MODEL_PATH, FEAT_NAMES_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artefact not found: '{path}'.\n"
                "Please run  src/train.py  first."
            )

    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_NAMES_PATH)
    return model, feature_names


def find_optimal_price(
    demand: float,
    season: str,
    competitor_price: float,
    discount: float = 10.0,
    category_num: int = 0,
) -> dict:
    """
    Sweep candidate prices and return the price that maximises predicted sales.

    Parameters
    ----------
    demand           : current demand level (20–300)
    season           : one of 'summer', 'winter', 'spring', 'fall'
    competitor_price : competitor's current price (₹)
    discount         : discount percentage applied (0–30)
    category_num     : encoded product category (default 0 = electronics)

    Returns
    -------
    dict with keys:
        optimal_price   (float) – best price found
        predicted_sales (int)   – predicted units sold at optimal_price
        increase_percent(float) – sales lift vs. the mid-range baseline price
    """
    # Validate season input
    season = season.lower().strip()
    if season not in SEASON_MAP:
        raise ValueError(f"Unknown season '{season}'. Choose from {list(SEASON_MAP)}")

    season_num = SEASON_MAP[season]

    # Load model (lazy — fast on repeated calls because joblib caches)
    model, feature_names = load_model()

    # Build a DataFrame row for every candidate price
    candidate_prices = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP, PRICE_STEP)

    import pandas as pd
    rows = pd.DataFrame({
        "current_price":    candidate_prices,
        "demand":           demand,
        "season_num":       season_num,
        "competitor_price": competitor_price,
        "discount":         discount,
        "category_num":     category_num,
    })[feature_names]   # reorder to match training column order

    # Predict sales for all prices in one batch (fast)
    predicted_sales = model.predict(rows)

    # Pick the price with the highest predicted revenue.
    predicted_revenue = candidate_prices * predicted_sales
    best_idx          = int(np.argmax(predicted_revenue))
    optimal_price     = float(candidate_prices[best_idx])
    best_sales        = float(predicted_sales[best_idx])
    best_revenue      = float(predicted_revenue[best_idx])

    # Baseline: revenue at the mid-range price.
    mid_idx          = len(candidate_prices) // 2
    baseline_revenue = float(predicted_revenue[mid_idx])
    increase_percent = round(
        ((best_revenue - baseline_revenue) / max(baseline_revenue, 1)) * 100, 1
    )

    return {
        "optimal_price":    optimal_price,
        "predicted_sales":  int(round(best_sales)),
        "predicted_revenue": round(best_revenue, 2),
        "increase_percent": increase_percent,
    }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running prediction smoke-test...\n")

    test_cases = [
        {"demand": 150, "season": "summer", "competitor_price": 450, "discount": 10},
        {"demand": 80, "season": "winter", "competitor_price": 700, "discount": 20},
        {"demand": 250, "season": "spring", "competitor_price": 300, "discount": 5},
    ]

    try:
        for tc in test_cases:
            result = find_optimal_price(**tc)
            print(f"  Input  : {tc}")
            print(
                f"  Output : Optimal Rs. {result['optimal_price']:.0f}  |  "
                f"Sales {result['predicted_sales']} units  |  "
                f"Revenue Rs. {result['predicted_revenue']:,.0f}  |  "
                f"+{result['increase_percent']}% revenue lift\n"
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Smoke-test passed!")
