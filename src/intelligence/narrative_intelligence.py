import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.utils.data_loader import load_master_dataset


# --------------------------------------------------
# Narrative Attribution
# --------------------------------------------------

def run_narrative_attribution(df):

    topic_brand = (
        df.groupby(["topic", "brand"])
        .size()
        .reset_index(name="count")
    )

    dominant = (
        topic_brand.sort_values("count", ascending=False)
        .groupby("topic")
        .head(2)
    )

    attribution = {}

    for topic in dominant["topic"].unique():

        brands = dominant[dominant["topic"] == topic]["brand"].tolist()

        attribution[topic] = brands

    return attribution


# --------------------------------------------------
# Topic × Brand Sentiment Matrix
# --------------------------------------------------

def run_topic_brand_sentiment_matrix(df):

    mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["finbert_label"].map(mapping)

    matrix = (
        df.groupby(["topic", "brand"])["sentiment_score"]
        .mean()
        .reset_index()
    )

    result = []

    for _, row in matrix.iterrows():

        result.append({
            "topic": row["topic"],
            "brand": row["brand"],
            "avg_sentiment": round(float(row["sentiment_score"]), 3)
        })

    return result


# --------------------------------------------------
# Narrative Risk Detector
# --------------------------------------------------

def run_narrative_risk_detector(df):

    mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["finbert_label"].map(mapping)

    topic_sentiment = (
        df.groupby("topic")["sentiment_score"]
        .mean()
        .sort_values()
    )

    risk_topics = topic_sentiment.head(3)

    risk_output = []

    for topic, score in risk_topics.items():

        affected_brands = (
            df[df["topic"] == topic]["brand"]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )

        risk_output.append({
            "topic": topic,
            "avg_sentiment": round(float(score), 3),
            "affected_brands": affected_brands
        })

    return risk_output


# --------------------------------------------------
# Master Narrative Intelligence Runner
# --------------------------------------------------

def run_narrative_intelligence():

    print("Running Narrative Intelligence Engine...")

    df = load_master_dataset()

    attribution = run_narrative_attribution(df)

    topic_brand_matrix = run_topic_brand_sentiment_matrix(df)

    narrative_risks = run_narrative_risk_detector(df)

    impact_scores = run_topic_impact_score(df)

    result = {
        "narrative_attribution": attribution,
        "topic_brand_sentiment_matrix": topic_brand_matrix,
        "narrative_risks": narrative_risks,
        "topic_impact_scores": impact_scores
    }

    print("\n--- Narrative Attribution ---")
    print(attribution)

    print("\n--- Narrative Risk Topics ---")
    print(narrative_risks)

    return result

def run_topic_impact_score(df):

    mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["finbert_label"].map(mapping)

    # Topic sentiment strength
    topic_sentiment = (
        df.groupby("topic")["sentiment_score"]
        .mean()
    )

    # Topic volume
    topic_volume = df["topic"].value_counts()

    impact_scores = []

    for topic in topic_sentiment.index:

        sentiment_strength = abs(topic_sentiment[topic])
        volume_weight = topic_volume[topic] / len(df)

        impact = sentiment_strength * volume_weight

        impact_scores.append({
            "topic": topic,
            "avg_sentiment": round(float(topic_sentiment[topic]), 3),
            "volume": int(topic_volume[topic]),
            "impact_score": round(float(impact), 3)
        })

    impact_scores = sorted(
        impact_scores,
        key=lambda x: x["impact_score"],
        reverse=True
    )

    return impact_scores
if __name__ == "__main__":
    run_narrative_intelligence()