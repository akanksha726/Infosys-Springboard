import pandas as pd
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.consumer.consumer_sentiment import analyze_reviews
from src.rag.rag_engine import generate_rag_response


def generate_consumer_insights(BASE_DIR):
    reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews_dataset.csv")
    consumer_path = os.path.join(BASE_DIR, "data", "processed", "consumer_sentiment.csv")
    finbert_summary_path = os.path.join(BASE_DIR, "data", "processed", "finbert_summary.json")

    # default empty distribution
    sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}

    # -------------------------
    # STEP 1: RUN SENTIMENT (if reviews exist)
    # -------------------------
    if os.path.exists(reviews_path):
        print("Running consumer sentiment...")
        analyze_reviews(reviews_path, consumer_path)
    else:
        print("⚠️ No reviews data found, skipping sentiment generation")

    # -------------------------
    # STEP 2: LOAD CONSUMER DATA OR FALLBACK
    # -------------------------
    if not os.path.exists(consumer_path):
        print("⚠️ Consumer sentiment file not found. Falling back to news sentiment distribution.")
        
        # Fallback to finbert_summary.json for the distribution chart
        if os.path.exists(finbert_summary_path):
            try:
                with open(finbert_summary_path, "r") as f:
                    fin_summary = json.load(f)
                    sentiment_distribution = {
                        "positive": fin_summary.get("positive_count", 0),
                        "negative": fin_summary.get("negative_count", 0),
                        "neutral": fin_summary.get("neutral_count", 0)
                    }
                print(f"✅ Loaded distribution from news: {sentiment_distribution}")
            except Exception as e:
                print(f"⚠️ Failed to load news sentiment fallback: {e}")

        return {
            "brand_sentiment": [],
            "top_positive_brands": [],
            "top_negative_brands": [],
            "sentiment_distribution": sentiment_distribution,
            "consumer_ai_insight": "Market sentiment currently driven primarily by news cycles due to limited review data availability."
        }

    # -------------------------
    # STEP 3: PROCESS EXISTING CONSUMER DATA
    # -------------------------
    consumer_df = pd.read_csv(consumer_path)

    consumer_df["brand"] = consumer_df["brand"].str.strip().str.lower()
    consumer_df = consumer_df[consumer_df["brand"] != "other"]

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

    # distribution from reviews
    sentiment_distribution = {
        "positive": int((consumer_df["raw_sentiment"] > 0.05).sum()),
        "negative": int((consumer_df["raw_sentiment"] < -0.05).sum()),
        "neutral": int(((consumer_df["raw_sentiment"] >= -0.05) & (consumer_df["raw_sentiment"] <= 0.05)).sum())
    }

    # AI insight
    try:
        print("Generating consumer sentiment AI insight...")

        consumer_summary = f"""
Top positive brands by consumer sentiment: {", ".join([f"{b['brand']} ({b['final_consumer_sentiment']:.2f})" for b in top_positive])}
Top negative brands by consumer sentiment: {", ".join([f"{b['brand']} ({b['final_consumer_sentiment']:.2f})" for b in top_negative])}

Explain the specific market signals or news events behind these consumer sentiment trends. 
Compare the top positive and top negative brands.
Identify if there's a dominant market topic (like discounts, logistics, or expansion) driving these trends.
"""

        consumer_ai_insight, sources = generate_rag_response(consumer_summary, k=12)

    except Exception as e:
        print("⚠️ RAG failed:", e)
        consumer_ai_insight = ""
        sources = []

    return {
        "brand_sentiment": consumer_data,
        "top_positive_brands": top_positive,
        "top_negative_brands": top_negative,
        "sentiment_distribution": sentiment_distribution,
        "consumer_ai_insight": consumer_ai_insight,
        "sources": sources
    }