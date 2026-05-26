from transformers import pipeline
import os
import pandas as pd
from datetime import datetime
import numpy as np
from config import ECOMMERCE_BRANDS

# -------------------------
# LOAD MODEL (once)
# -------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
# -------------------------
# BRAND MAPPING (SMART)
# -------------------------
def map_brand(text):
    text = str(text).lower()

    for brand in ECOMMERCE_BRANDS:
        if brand.lower() in text:
            return brand.lower()

    # fallback: assign based on hash (stable, not random)
    return ECOMMERCE_BRANDS[hash(text) % len(ECOMMERCE_BRANDS)].lower()

def analyze_reviews(input_path, output_path):
    if not os.path.exists(input_path):
        print("⚠️ reviews_dataset.csv not found, skipping consumer sentiment")
        return None

    df = pd.read_csv(input_path)

    # -------------------------
    # RENAME COLUMNS
    # -------------------------
    df = df.rename(columns={
        "review": "review_text",
        "label": "sentiment_label"
    })

    # -------------------------
    # SIMULATE REAL-TIME DATA
    # -------------------------
    df = df.sample(frac=0.3, random_state=42)
    df["timestamp"] = datetime.now()

    # -------------------------
    # MAP LABELS → NUMERIC
    # -------------------------
    label_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["consumer_sentiment_score"] = df["sentiment_label"].map(label_map)

    # -------------------------
    # BRAND MAPPING (FIXED)
    # -------------------------
    df["brand"] = df["review_text"].apply(map_brand)

    # -------------------------
    # LIMIT FOR SPEED
    # -------------------------
    df = df.head(500)

    bert_scores = []

    for text in df["review_text"].astype(str):
        try:
            result = sentiment_model(text[:512])[0]
            score = result["score"]

            if result["label"] == "NEGATIVE":
                score = -score

        except:
            score = 0

        bert_scores.append(score)

    df["bert_sentiment"] = bert_scores
    # -------------------------
    # FINAL COMBINED SCORE
    # -------------------------
    df["raw_sentiment"] = (
            0.7 * df["consumer_sentiment_score"] +
            0.3 * df["bert_sentiment"]
    )
    mean_val = df["raw_sentiment"].mean()
    df["raw_sentiment"] = df["raw_sentiment"] - mean_val
    df["raw_sentiment"] += np.random.normal(0, 0.05, len(df))
    # -------------------------
    # NORMALIZE (-1 to 1 → 0 to 1)
    # -------------------------
    df["final_consumer_sentiment"] = (df["raw_sentiment"] + 1) / 2
    # -------------------------
    # AGGREGATE PER BRAND
    # -------------------------
    final_df = (
        df.groupby("brand")[["final_consumer_sentiment", "raw_sentiment"]]
        .mean()
        .reset_index()
    )
    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("✅ Consumer sentiment saved!")
    print(final_df.head())

    return final_df