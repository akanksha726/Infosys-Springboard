import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.utils.data_loader import load_master_dataset


def run_topic_momentum_tracker():

    print("Running Topic Momentum Tracker...")

    df = load_master_dataset()

    if df.empty:
        return {"error": "Master dataset empty"}

    # -----------------------------
    # Prepare dates
    # -----------------------------

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.date

    # -----------------------------
    # Daily topic counts
    # -----------------------------

    topic_daily = (
        df.groupby(["date", "topic"])
        .size()
        .reset_index(name="count")
    )

    topic_pivot = topic_daily.pivot(
        index="date",
        columns="topic",
        values="count"
    ).fillna(0)

    if len(topic_pivot) < 2:
        return {"error": "Not enough time data for momentum"}

    # -----------------------------
    # Compute momentum
    # -----------------------------

    topic_momentum = {}

    for topic in topic_pivot.columns:

        series = topic_pivot[topic]

        momentum = series.iloc[-1] - series.iloc[-2]

        topic_momentum[topic] = int(momentum)

    # -----------------------------
    # Sort topics
    # -----------------------------

    rising_topics = sorted(
        topic_momentum.items(),
        key=lambda x: x[1],
        reverse=True
    )

    declining_topics = sorted(
        topic_momentum.items(),
        key=lambda x: x[1]
    )

    top_rising = rising_topics[:3]
    top_declining = declining_topics[:3]

    # -----------------------------
    # Format results
    # -----------------------------

    rising_output = [
        {"topic": topic, "momentum": momentum}
        for topic, momentum in top_rising
    ]

    declining_output = [
        {"topic": topic, "momentum": momentum}
        for topic, momentum in top_declining
    ]

    result = {
        "top_rising_topics": rising_output,
        "top_declining_topics": declining_output
    }

    print("\n--- Topic Momentum ---")

    print("\nTop Rising Topics:")
    for t in rising_output:
        print(f"{t['topic']} (Δ {t['momentum']})")

    print("\nTop Declining Topics:")
    for t in declining_output:
        print(f"{t['topic']} (Δ {t['momentum']})")

    return result


if __name__ == "__main__":
    run_topic_momentum_tracker()