import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.data_loader import (
    load_master_dataset,
    load_daily_metrics,
    load_brand_metrics
)
from src.rag.rag_engine import generate_rag_response


def run_market_intelligence():

    print("Running Market Intelligence Engine...")

    # -------------------------
    # Load Master Dataset
    # -------------------------

    master_df = load_master_dataset()

    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df["date"] = master_df["date"].dt.date

    # -------------------------
    # 1️⃣ Topic Trend Intelligence
    # -------------------------

    topic_daily = (
        master_df.groupby(["date", "topic"])
        .size()
        .reset_index(name="count")
    )

    topic_pivot = topic_daily.pivot(
        index="date",
        columns="topic",
        values="count"
    ).fillna(0)

    topic_momentum = {}

    for topic in topic_pivot.columns:

        series = topic_pivot[topic]

        if len(series) < 2:
            continue

        momentum = series.iloc[-1] - series.iloc[-2]
        topic_momentum[topic] = momentum

    latest_topics = topic_pivot.iloc[-1].sort_values(ascending=False)
    top_topics = latest_topics.head(3).index.tolist()

    if topic_momentum:
        fastest_rising_topic = max(topic_momentum, key=topic_momentum.get)
        fastest_declining_topic = min(topic_momentum, key=topic_momentum.get)
    else:
        fastest_rising_topic = None
        fastest_declining_topic = None

    # -------------------------
    # 2️⃣ Topic Sentiment Analysis
    # -------------------------

    mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    master_df["sentiment_score"] = master_df["finbert_label"].map(mapping)

    topic_sentiment = (
        master_df.groupby("topic")["sentiment_score"]
        .mean()
        .sort_values(ascending=False)
    )

    most_positive_topic = topic_sentiment.index[0]
    most_negative_topic = topic_sentiment.index[-1]

    # -------------------------
    # 3️⃣ Market Trend
    # -------------------------

    daily_df = load_daily_metrics()

    # -------------------------
    # 🔥 MERGE FINAL TREND INTO DAILY DATA (NEW)
    # -------------------------

    # convert dates to datetime
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    master_df["date"] = pd.to_datetime(master_df["date"])

    # aggregate final trend from master dataset
    trend_daily = (
        master_df.groupby("date")["final_trend_score"]
        .mean()
        .reset_index()
    )

    # merge into daily_df
    daily_df = pd.merge(
        daily_df,
        trend_daily,
        on="date",
        how="left"
    )

    # fill missing values
    daily_df["final_trend_score"] = daily_df["final_trend_score"].fillna(0)

    daily_df = daily_df.sort_values("date")
    daily_df["time_index"] = range(len(daily_df))

    X = daily_df[["time_index"]]
    y = daily_df["final_trend_score"]

    model = LinearRegression()
    model.fit(X, y)

    market_slope = model.coef_[0]
    current_sentiment = daily_df["final_trend_score"].iloc[-1]

    if market_slope > 0:
        market_direction = "Bullish"
    elif market_slope < 0:
        market_direction = "Bearish"
    else:
        market_direction = "Stable"

    # -------------------------
    # 4️⃣ Brand Momentum
    # -------------------------

    brand_df = load_brand_metrics()

    momentum_results = []

    for brand in brand_df["brand"].unique():

        temp = brand_df[brand_df["brand"] == brand].sort_values("date")

        if len(temp) < 2:
            continue

        temp["time_index"] = range(len(temp))

        Xb = temp[["time_index"]]
        yb = temp["sentiment_index"]

        model = LinearRegression()
        model.fit(Xb, yb)

        slope = model.coef_[0]

        momentum_results.append((brand, slope))

    momentum_results = sorted(momentum_results, key=lambda x: x[1], reverse=True)

    top_positive_brand = momentum_results[0][0]
    top_negative_brand = momentum_results[-1][0]

    # -------------------------
    # 5️⃣ Brand Volatility
    # -------------------------

    volatility_results = []

    for brand in brand_df["brand"].unique():

        temp = brand_df[brand_df["brand"] == brand]

        if len(temp) < 2:
            continue

        volatility = temp["sentiment_index"].std()
        volatility_results.append((brand, volatility))

    volatility_results = sorted(volatility_results, key=lambda x: x[1], reverse=True)

    most_volatile_brand = volatility_results[0][0]

    # -------------------------
    # 6️⃣ Narrative Summary
    # -------------------------

    summary = f"""
    Overall ecommerce market sentiment is currently {market_direction.lower()}
    with a slope of {round(market_slope,4)}.

    Current sentiment score stands at {round(current_sentiment,3)}.

    Brand momentum analysis shows {top_positive_brand} gaining the strongest positive sentiment momentum,
    while {top_negative_brand} is experiencing the sharpest decline.

    Sentiment volatility is highest for {most_volatile_brand}.

    Top narrative topics today include {top_topics}.

    The most positively perceived topic is {most_positive_topic},
    while the most negative narrative is around {most_negative_topic}.

    The fastest rising topic is {fastest_rising_topic},
    while the fastest declining topic is {fastest_declining_topic}.

    """

    # -------------------------
    # 🔮 7️⃣ RAG Narrative Insight
    # -------------------------

    print("Generating RAG market explanation...")

    rag_query = f"""
Explain the current Indian ecommerce market sentiment.

Market Direction: {market_direction}
Top Positive Brand: {top_positive_brand}
Top Negative Brand: {top_negative_brand}
Top Topics: {top_topics}
Fastest Rising Topic: {fastest_rising_topic}
Fastest Declining Topic: {fastest_declining_topic}

Provide a concise explanation of why these trends may be happening.
"""

    rag_insight, rag_sources = generate_rag_response(rag_query)

    summary += f"""

    AI Narrative Explanation
    {rag_insight}
    """

    output = {
        "market_direction": market_direction,
        "market_slope": round(float(market_slope), 6),
        "current_sentiment": round(float(current_sentiment), 6),
        "top_positive_brand": top_positive_brand,
        "top_negative_brand": top_negative_brand,
        "most_volatile_brand": most_volatile_brand,
        "top_topics": top_topics,
        "most_positive_topic": most_positive_topic,
        "most_negative_topic": most_negative_topic,
        "fastest_rising_topic": fastest_rising_topic,
        "fastest_declining_topic": fastest_declining_topic,
        "rag_market_explanation": rag_insight,
        "rag_sources": rag_sources
    }

    print("\n--- Human Readable Insight ---")
    print(summary)

    print("\n--- JSON Output ---")
    print(json.dumps(output, indent=4))

    return output, summary