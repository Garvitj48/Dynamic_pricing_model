#!/usr/bin/env python3
"""
src/train.py
------------
Trains a Random Forest Regressor on the e-commerce pricing dataset and
saves the fitted model + feature names for use by the prediction module.

Run from the project root:
    python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH         = os.path.join("data", "online_retail.csv")
MODEL_DIR         = "models"
MODEL_PATH        = os.path.join(MODEL_DIR, "pricing_model.pkl")
FEAT_NAMES_PATH   = os.path.join(MODEL_DIR, "feature_names.pkl")

# Features the model is trained on (must match predict.py)
FEATURE_COLS = [
    "current_price",
    "demand",
    "season_num",
    "competitor_price",
    "discount",
    "category_num",
]
TARGET_COL = "sales_volume"


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset and validate required columns exist."""
    if not os.path.exists(path):
        print(f" Dataset not found at '{path}'.")
        print("   Please run  mock_data.py  first to generate the dataset.")
        sys.exit(1)

    df = pd.read_csv(path)
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns in dataset: {missing}")
        sys.exit(1)

    print(f" Loaded dataset: {len(df):,} rows, {df.shape[1]} columns")
    return df


def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame  – training features
    y_train : pd.Series     – training target (sales_volume)

    Returns
    -------
    Fitted RandomForestRegressor
    """
    print("\n Training Random Forest Regressor …")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(" Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """Print MAE and R² on the held-out test set."""
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print("\n── Model Performance ──────────────────────────────")
    print(f"   MAE  (Mean Absolute Error) : {mae:.2f} units")
    print(f"   R²   (Coefficient of Det.) : {r2:.4f}")
    print("───────────────────────────────────────────────────")
    return mae, r2


def print_feature_importance(model, feature_names: list):
    """Display a ranked table of feature importances."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n── Feature Importances ────────────────────────────")
    print(f"   {'Feature':<20} Importance")
    print(f"   {'-'*20} ----------")
    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"   {feat:<20} {imp:.4f}  {bar}")
    print("───────────────────────────────────────────────────")


def save_artifacts(model, feature_names: list):
    """Persist the trained model and feature name list to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(feature_names, FEAT_NAMES_PATH)
    print(f"\n Model saved       → {MODEL_PATH}")
    print(f" Feature names saved → {FEAT_NAMES_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare features & target
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 3. Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

    # 6. Feature importance
    print_feature_importance(model, FEATURE_COLS)

    # 7. Save
    save_artifacts(model, FEATURE_COLS)

    print("\n All done! Run  app.py  to launch the web interface.")
