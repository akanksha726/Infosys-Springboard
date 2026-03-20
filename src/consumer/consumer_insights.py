import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.consumer.consumer_sentiment import analyze_reviews
from src.rag.rag_engine import generate_rag_response


def generate_consumer_insights(BASE_DIR):
    reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews_data.csv")
    consumer_path = os.path.join(BASE_DIR, "data", "processed", "consumer_sentiment.csv")

    # -------------------------
    # STEP 1: RUN SENTIMENT (if reviews exist)
    # -------------------------
    if os.path.exists(reviews_path):
        print("Running consumer sentiment...")
        analyze_reviews(reviews_path, consumer_path)
    else:
        print("⚠️ No reviews data found, skipping sentiment generation")

    if not os.path.exists(consumer_path):
        print("⚠️ Consumer sentiment file not found")
        return {
            "brand_sentiment": [],
            "top_positive_brands": [],
            "top_negative_brands": [],
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "consumer_ai_insight": ""
        }

    consumer_df = pd.read_csv(consumer_path)

    consumer_df["brand"] = consumer_df["brand"].str.strip().str.lower()

    # sort
    consumer_df = consumer_df.sort_values(
        "final_consumer_sentiment",
        ascending=False
    )

    consumer_data = consumer_df.to_dict(orient="records")

    # top brands
    top_positive = consumer_df.head(3).to_dict(orient="records")

    top_negative = consumer_df.sort_values(
        "final_consumer_sentiment", ascending=True
    ).head(3).to_dict(orient="records")

    # distribution
    sentiment_distribution = {
        "positive": int((consumer_df["final_consumer_sentiment"] > 0).sum()),
        "negative": int((consumer_df["final_consumer_sentiment"] < 0).sum()),
        "neutral": int((consumer_df["final_consumer_sentiment"] == 0).sum())
    }

    # AI insight
    try:
        print("Generating consumer sentiment AI insight...")

        consumer_summary = f"""
Top positive brands: {", ".join([b['brand'] for b in top_positive])}
Top negative brands: {", ".join([b['brand'] for b in top_negative])}

Explain why these brands might have such sentiment trends.
"""

        consumer_ai_insight = generate_rag_response(consumer_summary)

    except Exception as e:
        print("⚠️ RAG failed:", e)
        consumer_ai_insight = ""

    return {
        "brand_sentiment": consumer_data,
        "top_positive_brands": top_positive,
        "top_negative_brands": top_negative,
        "sentiment_distribution": sentiment_distribution,
        "consumer_ai_insight": consumer_ai_insight
    }