#!/usr/bin/env python3
"""
mock_data.py
------------
Generates a synthetic e-commerce dataset with 10,000 products for training
the Dynamic Pricing Model. Run this script once before training.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 10_000  # number of synthetic product records

# ── 1. Raw feature generation ─────────────────────────────────────────────────
print("⏳ Generating synthetic e-commerce dataset …")

product_id       = [f"P{str(i).zfill(5)}" for i in range(1, N + 1)]
current_price    = np.random.randint(199, 1000, size=N).astype(float)
demand           = np.random.randint(20,  301,  size=N).astype(float)
seasons          = np.random.choice(["summer", "winter", "spring", "fall"], size=N)
competitor_price = np.random.randint(150, 951,  size=N).astype(float)
categories       = np.random.choice(
    ["electronics", "clothing", "accessories", "home"], size=N
)
discount         = np.random.randint(0, 31, size=N).astype(float)  # 0–30 %

# ── 2. Encode categoricals ────────────────────────────────────────────────────
season_map   = {"summer": 0, "winter": 1, "spring": 2, "fall": 3}
category_map = {"electronics": 0, "clothing": 1, "accessories": 2, "home": 3}

season_num   = np.array([season_map[s]   for s in seasons])
category_num = np.array([category_map[c] for c in categories])

# ── 3. Realistic sales-volume formula ────────────────────────────────────────
#  Higher demand  → more sales
#  Lower price    → more sales
#  Higher discount → more sales
#  Closer to competitor price (or below) → more sales
#  Some Gaussian noise to make it realistic
price_effect      = np.clip(1 - (current_price / 1200), 0.1, 1.0)   # [0.1, 1.0]
demand_effect     = demand / demand.max()                             # [0, 1]
discount_effect   = 1 + (discount / 100) * 0.5                       # [1.0, 1.15]
competitor_effect = np.clip(
    1 + (competitor_price - current_price) / competitor_price, 0.5, 1.5
)
noise = np.random.normal(0, 5, size=N)

sales_volume = (
    demand_effect * 200
    * price_effect
    * discount_effect
    * competitor_effect
    + noise
).clip(1).round().astype(int)

# ── 4. Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "product_id":       product_id,
    "current_price":    current_price,
    "demand":           demand,
    "season":           seasons,
    "season_num":       season_num,
    "competitor_price": competitor_price,
    "category":         categories,
    "category_num":     category_num,
    "discount":         discount,
    "sales_volume":     sales_volume,
})

# ── 5. Save to CSV ────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "online_retail.csv")
df.to_csv(output_path, index=False)

print(f"  Dataset saved to '{output_path}'")
print(f"   Rows : {len(df):,}")
print(f"   Cols : {list(df.columns)}")
print(f"\nSample rows:\n{df.head(3).to_string(index=False)}")
