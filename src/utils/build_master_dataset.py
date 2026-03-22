import pandas as pd
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

from src.analytics.feature_engineering import create_features

def build_master_dataset():

    cleaned_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_news.csv")
    finbert_path = os.path.join(BASE_DIR, "data", "processed", "finbert_output.csv")
    topic_path = os.path.join(BASE_DIR, "data", "processed", "news_with_topics.csv")

    cleaned_df = pd.read_csv(cleaned_path)
    finbert_df = pd.read_csv(finbert_path)
    topic_df = pd.read_csv(topic_path)
    # 🔥 LOAD RAW NEWS (for title + url)
    raw_news_path = os.path.join(BASE_DIR, "data", "raw", "news_data.csv")
    raw_df = pd.read_csv(raw_news_path)

    # keep only required columns
    raw_df = raw_df[["title", "url"]]

    master_df = cleaned_df.copy()

    # 🔥 ADD TITLE + URL (SAFE - index aligned)
    if len(raw_df) == len(master_df):
        master_df["title"] = raw_df["title"]
        master_df["url"] = raw_df["url"]
    else:
        print("⚠️ Warning: raw_df size mismatch — skipping title/url merge")
        master_df["title"] = ""
        master_df["url"] = ""

    master_df["title"] = master_df["title"].fillna("")
    master_df["url"] = master_df["url"].fillna("")

    # -------------------------
    # ADD SENTIMENT + TOPIC
    # -------------------------

    master_df["finbert_label"] = finbert_df["finbert_label"]
    master_df["finbert_confidence"] = finbert_df["finbert_confidence"]

    master_df["topic"] = topic_df["topic"]
    # ---------------------------
    # 🔥 ADD BACK SENTIMENT FEATURES (CRITICAL)
    # ---------------------------

    master_df["topic_confidence"] = topic_df["topic_confidence"]

    SENTIMENT_MAP = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    master_df["sentiment_score"] = master_df["finbert_label"].str.lower().map(SENTIMENT_MAP)

    master_df["weighted_sentiment"] = (
            master_df["sentiment_score"] * master_df["finbert_confidence"]
    )
    # ---------------------------
    # 🔥 CREATE TOPIC FLAGS
    # ---------------------------
    topic_list = [
        "funding", "discounts", "technology", "logistics",
        "customer_complaints", "expansion", "competition",
        "partnership", "other"
    ]

    for t in topic_list:
        master_df[t] = (master_df["topic"] == t).astype(int)

    # -------------------------
    # FIX COLUMN NAMES
    # -------------------------

    master_df = master_df.rename(columns={
        "Published_Date": "date",
        "Brand": "brand"
    })

    # ---------------------------
    # 🔥 REAL Topic Concentration (Entropy-based)
    # ---------------------------

    import numpy as np

    # Ensure date exists
    master_df["date"] = pd.to_datetime(master_df["date"], errors="coerce").dt.date

    # Step 1: count topics per day
    topic_counts = master_df.groupby(["date", "topic"]).size()

    # Step 2: convert to probability
    topic_prob = topic_counts / topic_counts.groupby(level=0).sum()

    # Step 3: entropy per day
    topic_entropy = - (topic_prob * np.log(topic_prob + 1e-9)).groupby(level=0).sum()

    # Step 4: convert entropy → concentration
    topic_concentration = 1 / (1 + topic_entropy)

    # Step 5: merge back
    topic_concentration_df = topic_concentration.reset_index()
    topic_concentration_df.columns = ["date", "topic_concentration"]

    master_df = master_df.merge(
        topic_concentration_df,
        on="date",
        how="left"
    )
    master_df["topic_concentration"] = ((master_df["topic_concentration"] - master_df["topic_concentration"].min())
                                        / (
                                            master_df["topic_concentration"].max() - master_df["topic_concentration"].min() + 1e-6
                                       ))
    # fallback
    master_df["topic_concentration"] = master_df["topic_concentration"].fillna(0.5)


    master_df['date'] = pd.to_datetime(
        master_df['date'],
        errors='coerce',
        utc=True  # 🔥 important fix
    )

    # drop bad rows
    master_df = master_df.dropna(subset=['date'])

    # convert safely
    master_df['date'] = master_df['date'].dt.tz_localize(None)
    master_df['brand'] = master_df['brand'].str.strip().str.lower()
    master_df = master_df.sort_values(["brand", "date"])
    master_df.columns = master_df.columns.str.lower()

    # =========================================================
    # 🔥 ADD EXTERNAL TREND DATA (NEW)
    # =========================================================

    trend_path = os.path.join(BASE_DIR, "data", "processed", "trend_data_cleaned.csv")
    trend_df = pd.read_csv(trend_path)

    # ---------------------------
    # FIX DATE PARSING (FINAL)
    # ---------------------------
    trend_df['date'] = pd.to_datetime(trend_df['date'], errors='coerce')
    trend_df['date'] = trend_df['date'].dt.normalize()
    # since already midpoint date → treat as single-day signal
    trend_df['start_date'] = trend_df['date']
    trend_df['end_date'] = trend_df['date']

    print("After parsing:")
    print(trend_df[['start_date', 'end_date']].head())
    # ---------------------------
    # EXPAND TO DAILY
    # ---------------------------

    trend_df = trend_df.rename(columns={
        "trend_score": "external_trend_score"
    })

    trend_df = trend_df[["date", "brand", "external_trend_score"]]

    print("After parsing:")
    # clean trend data
    trend_df['brand'] = trend_df['brand'].str.strip().str.lower()

    # merge external trend
    master_df = pd.merge(
        master_df,
        trend_df[['date', 'brand', 'external_trend_score']],
        on=['date', 'brand'],
        how='left'
    )

    # fill missing external trends
    master_df['external_trend_score'] = master_df['external_trend_score'].fillna(0)

    # =========================================================
    # 🔥 INTERNAL TREND ENGINE (FINAL)
    # =========================================================

    # base trend from sentiment
    master_df["trend_score"] = (master_df["weighted_sentiment"] + 1) / 2

    # smooth per brand
    master_df["trend_score"] = master_df.groupby("brand")["trend_score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # velocity
    master_df["trend_velocity"] = master_df.groupby("brand")["trend_score"].diff()

    # acceleration
    master_df["trend_acceleration"] = master_df.groupby("brand")["trend_velocity"].diff()

    # combined signal
    master_df["trend_sentiment_signal"] = (
            master_df["trend_velocity"] * master_df["weighted_sentiment"]
    )

    # fill nulls
    master_df[[
        "trend_velocity",
        "trend_acceleration",
        "trend_sentiment_signal"
    ]] = master_df[[
        "trend_velocity",
        "trend_acceleration",
        "trend_sentiment_signal"
    ]].fillna(0)

    # =========================================================
    # 🔥 FINAL HYBRID SCORE (NEW)
    # =========================================================

    master_df["final_trend_score"] = (
        0.6 * master_df["external_trend_score"] +
        0.4 * (master_df["trend_score"])
    )

    # -------------------------
    # SAVE DATASET
    # -------------------------

    output_path = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")

    master_df.to_csv(output_path, index=False)

    print(master_df[[
        "brand",
        "external_trend_score",
        "trend_score",
        "final_trend_score",
        "trend_velocity"
    ]].head(20))
    print("Master dataset created")
    print(output_path)