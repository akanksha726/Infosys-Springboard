import pandas as pd
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "featured_dataset.csv")


def create_features(df):

    print("Creating features...")

    # -------------------------
    # STEP 1: Ensure date format
    # -------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -------------------------
    # STEP 2: CREATE sentiment_index (CRITICAL FIX)
    # -------------------------
    daily_df = (
        df.groupby("date")["weighted_sentiment"]
        .mean()
        .reset_index(name="sentiment_index")
    )

    daily_df = daily_df.sort_values("date")

    # -------------------------
    # STEP 3: FEATURE ENGINEERING
    # -------------------------

    daily_df["sentiment_lag_1"] = daily_df["sentiment_index"].shift(1)
    daily_df["sentiment_lag_2"] = daily_df["sentiment_index"].shift(2)

    daily_df["rolling_mean_3"] = daily_df["sentiment_index"].rolling(3).mean()

    daily_df["sentiment_momentum"] = daily_df["sentiment_index"].diff()

    daily_df = daily_df.fillna(0)

    # -------------------------
    # SAVE (optional)
    # -------------------------
    daily_df.to_csv(OUTPUT_PATH, index=False)

    print("Feature dataset saved at:")
    print(OUTPUT_PATH)

    return daily_df

# -------------------------
# RUN DIRECTLY (OPTIONAL)
# -------------------------
if __name__ == "__main__":
    create_features(df=pd.read_csv(INPUT_PATH))