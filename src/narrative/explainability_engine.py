import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from src.utils.data_loader import load_master_dataset

def run_explainability_engine(base_dir):

    df = load_master_dataset()

    if df.empty:
        return {"error": "Dataset empty"}

    # -------------------------
    # Sentiment Mapping
    # -------------------------

    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["finbert_label"].map(sentiment_map)

    # -------------------------
    # Topic Sentiment Matrix
    # -------------------------

    topic_sentiment = (
        df.groupby("topic")["sentiment_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    topic_matrix = topic_sentiment.to_dict(orient="records")

    # -------------------------
    # Topic Explainability
    # -------------------------

    topic_examples = {}

    for topic in df["topic"].unique():

        topic_news = df[df["topic"] == topic]

        example_articles = topic_news["combined_text"].head(3).tolist()

        avg_sentiment = topic_news["sentiment_score"].mean()

        topic_examples[topic] = {
            "avg_sentiment": round(float(avg_sentiment), 3),
            "example_articles": example_articles
        }

    # -------------------------
    # Final Output
    # -------------------------

    explainability_output = {
        "topic_sentiment_matrix": topic_matrix,
        "topic_explainability": topic_examples
    }

    return explainability_output