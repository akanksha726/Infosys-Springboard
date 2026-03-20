import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = os.path.join(
    BASE_DIR, "..", "data", "processed", "trend_data_cleaned.csv"
)

INPUT_PATH = "../data/processed/trend_data.csv"

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(INPUT_PATH)

print("Original Data:")
print(df.head())

# ---------------------------
# CLEANING
# ---------------------------

# Convert trend_score to numeric
df['trend_score'] = pd.to_numeric(df['trend_score'], errors='coerce')

# Normalize (0–1 scale)
df['trend_score'] = df['trend_score'] / 100

df['trend_score'] = df['trend_score'].fillna(0)

# Convert date (optional)
df['date'] = df['date'].astype(str)

# ---------------------------
# OPTIONAL: Aggregate duplicates
# ---------------------------
df = df.groupby(['date', 'brand'], as_index=False)['trend_score'].mean()

# ---------------------------
# SAVE CLEAN DATA
# ---------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Cleaned data saved!")
print(df.head())