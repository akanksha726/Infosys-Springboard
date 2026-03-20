from transformers import pipeline
import os
import pandas as pd
import numpy as np
from config import ECOMMERCE_BRANDS

# -------------------------
# LOAD MODEL (once)
# -------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_reviews(input_path, output_path):
    if not os.path.exists(input_path):
        print("⚠️ reviews_data.csv not found, skipping consumer sentiment")
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
    # MAP LABELS → NUMERIC
    # -------------------------
    label_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["consumer_sentiment_score"] = df["sentiment_label"].map(label_map)

    # -------------------------
    # ADD BRAND (SIMULATED)
    # -------------------------
    brands = ECOMMERCE_BRANDS
    df["brand"] = np.random.choice(brands, len(df))

    # -------------------------
    # OPTIONAL: DistilBERT (limit for speed)
    # -------------------------
    df = df.head(500)

    bert_scores = []

    for text in df["review_text"].astype(str):
        result = sentiment_model(text[:512])[0]

        score = result["score"]

        if result["label"] == "NEGATIVE":
            score = -score

        bert_scores.append(score)

    df["bert_sentiment"] = bert_scores

    # -------------------------
    # FINAL COMBINED SCORE
    # -------------------------
    df["final_consumer_sentiment"] = (
        0.7 * df["consumer_sentiment_score"] +
        0.3 * df["bert_sentiment"]
    )

    # -------------------------
    # AGGREGATE PER BRAND
    # -------------------------
    final_df = (
        df.groupby("brand")["final_consumer_sentiment"]
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