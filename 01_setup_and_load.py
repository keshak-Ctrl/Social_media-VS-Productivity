"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 1: SETUP & LOAD     ║
╚══════════════════════════════════════════════════════════════╝
"""

import kagglehub
import pandas as pd
import os

# ── Download ──────────────────────────────────────────────────
path = kagglehub.dataset_download("mahdimashayekhi/social-media-vs-productivity")
csv_file = os.path.join(path, "social_media_vs_productivity.csv")

df = pd.read_csv(csv_file)

# ── Quick audit ───────────────────────────────────────────────
print(f"Shape     : {df.shape}")
print(f"Columns   : {list(df.columns)}")
print(f"Nulls     : {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(df.dtypes)

# ── Persist ───────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/social_media_vs_productivity.csv", index=False)
print("\n data/social_media_vs_productivity.csv")
