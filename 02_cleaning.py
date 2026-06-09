"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 2: DATA CLEANING    ║
╚══════════════════════════════════════════════════════════════╝
Pipeline:
  1. Missing-value indicators → KNN imputation
  2. 1–99th percentile winsorization
  3. Categorical standardisation
  4. Type correction
  5. Validation assertions
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("data/social_media_vs_productivity.csv")
print(f"Raw shape: {df.shape}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ── 1. Missing-value indicator flags ──────────────────────────
for col in numeric_cols:
    if df[col].isnull().any():
        df[f"{col}_was_missing"] = df[col].isnull().astype(int)
        print(f"  ↳ indicator added: {col}_was_missing")

# ── 2. KNN Imputation ─────────────────────────────────────────
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(f"\nPost-imputation nulls: {df.isnull().sum().sum()}")

# ── 3. Winsorization (1–99th percentile) ─────────────────────
WINSOR_COLS = [
    "daily_social_media_time", "number_of_notifications",
    "work_hours_per_day",       "perceived_productivity_score",
    "actual_productivity_score","stress_level",
    "sleep_hours",              "screen_time_before_sleep",
    "coffee_consumption_per_day","days_feeling_burnout_per_month",
    "weekly_offline_hours",     "job_satisfaction_score",
]

for col in WINSOR_COLS:
    if col not in df.columns:
        continue
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lo, hi)
    print(f"  ↳ winsorized {col}: [{lo:.2f}, {hi:.2f}]")

# ── 4. Standardise categorical columns ────────────────────────
CAT_COLS = ["gender", "job_type", "social_platform_preference"]
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# ── 5. Remove duplicates ──────────────────────────────────────
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicates removed: {before - len(df)}")

# ── 6. Correct data types ─────────────────────────────────────
for flag_col in ["uses_focus_apps", "has_digital_wellbeing_enabled"]:
    if flag_col in df.columns:
        df[flag_col] = df[flag_col].astype(int)

if "coffee_consumption_per_day" in df.columns:
    df["coffee_consumption_per_day"] = df["coffee_consumption_per_day"].round().astype(int)

# ── 7. Validation assertions ──────────────────────────────────
assert df["age"].between(10, 100).all(),                        "age OOB"
assert df["stress_level"].between(0, 10).all(),                 "stress OOB"
assert df["sleep_hours"].between(0, 24).all(),                  "sleep OOB"
assert df["perceived_productivity_score"].between(0, 10).all(), "perceived prod OOB"
assert df["actual_productivity_score"].between(0, 10).all(),    "actual prod OOB"
assert df["job_satisfaction_score"].between(0, 10).all(),       "satisfaction OOB"
print("\n All validation assertions passed")

# ── Save ──────────────────────────────────────────────────────
df.to_csv("data/cleaned_social_media.csv", index=False)
print(f"\n Saved → data/cleaned_social_media.csv")
print(f"  Final shape: {df.shape}\n")
