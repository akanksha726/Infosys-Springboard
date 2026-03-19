import pandas as pd
import os


def run_daily_sentiment_index():

    print("Calculating Daily Market Metrics...")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    input_path = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "daily_market_metrics.csv")

    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.date

    sentiment_mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["finbert_label"].map(sentiment_mapping)

    # -----------------------------
    # 1️⃣ Daily Sentiment Index
    # -----------------------------
    sentiment_index = (
        df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index(name="sentiment_index")
    )

    # -----------------------------
    # 2️⃣ Daily Topic Counts
    # -----------------------------
    topic_counts = (
        df.groupby(["date", "topic"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # -----------------------------
    # 3️⃣ Merge Sentiment + Topics
    # -----------------------------
    daily_metrics = pd.merge(
        sentiment_index,
        topic_counts,
        on="date",
        how="left"
    )

    daily_metrics.to_csv(output_path, index=False)

    print("Daily Market Metrics saved.")

    return output_path


if __name__ == "__main__":
    run_daily_sentiment_index()