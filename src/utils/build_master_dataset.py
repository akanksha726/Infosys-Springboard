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

    master_df = cleaned_df.copy()

    # -------------------------
    # ADD SENTIMENT + TOPIC
    # -------------------------

    master_df["finbert_label"] = finbert_df["finbert_label"]
    master_df["finbert_confidence"] = finbert_df["finbert_confidence"]

    master_df["topic"] = topic_df["topic"]
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

    # -------------------------
    # FIX COLUMN NAMES
    # -------------------------

    master_df = master_df.rename(columns={
        "Published_Date": "date",
        "Brand": "brand"
    })

    master_df['date'] = pd.to_datetime(master_df['date']).dt.date
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

    trend_df['date'] = trend_df['date'].str.replace(u'\u2009', '', regex=True).str.strip()

    # split into start + end
    trend_df[['start_part', 'end_part']] = trend_df['date'].str.split('–', expand=True)

    # extract year
    trend_df['year'] = trend_df['end_part'].str.extract(r'(\d{4})')

    # clean parts
    trend_df['start_part'] = trend_df['start_part'].str.strip()
    trend_df['end_part'] = trend_df['end_part'].str.replace(',', '').str.strip()

    # build full dates
    trend_df['start_date'] = pd.to_datetime(
        trend_df['start_part'] + " " + trend_df['year'],
        format="%b %d %Y",
        errors='coerce'
    )

    # extract month from start_part
    trend_df['month'] = trend_df['start_part'].str.split().str[0]

    # rebuild full end date with month
    trend_df['end_date'] = pd.to_datetime(
        trend_df['month'] + " " + trend_df['end_part'],
        format="%b %d %Y",
        errors='coerce'
    )

    print("After parsing:")
    print(trend_df[['start_date', 'end_date']].head())
    # ---------------------------
    # EXPAND TO DAILY
    # ---------------------------
    expanded_rows = []

    for _, row in trend_df.iterrows():
        if pd.isna(row['start_date']) or pd.isna(row['end_date']):
            continue

        date_range = pd.date_range(row['start_date'], row['end_date'])

        for d in date_range:
            expanded_rows.append({
                "date": d.date(),
                "brand": row['brand'],
                "external_trend_score": row['trend_score']
            })

    # safety check AFTER loop
    if not expanded_rows:
        print("❌ No expanded rows created!")
        print(trend_df[['date', 'start_date', 'end_date']].head())
        exit()

    trend_df = pd.DataFrame(expanded_rows)
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
    master_df["trend_score"] = (master_df["weighted_sentiment"] + 1) * 50

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
        0.4 * (master_df["trend_score"] / 100)
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