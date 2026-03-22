import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.utils.data_loader import load_master_dataset


def run_explainability_engine(base_dir):

    df = load_master_dataset()

    if df is None or df.empty:
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
    # 🔥 SAFE COLUMN HANDLING
    # -------------------------

    for col in ["title", "url", "brand", "topic", "weighted_sentiment"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("")
    df["url"] = df["url"].fillna("")
    df["brand"] = df["brand"].astype(str).str.lower()
    df["topic"] = df["topic"].astype(str).str.lower()

    if "weighted_sentiment" not in df.columns:
        df["weighted_sentiment"] = df["sentiment_score"]

    df["weighted_sentiment"] = df["weighted_sentiment"].fillna(0)

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
            "avg_sentiment": round(float(avg_sentiment), 3) if pd.notna(avg_sentiment) else 0,
            "example_articles": example_articles
        }

    # =========================================================
    # 🔥 SENTIMENT CHANGE ANALYSIS (TOP 3 REASONS)
    # =========================================================

    # -------------------------
    # Detect sentiment change
    # -------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    daily_sentiment = (
        df.groupby("date")["weighted_sentiment"]
        .mean()
        .sort_index()
    )

    if len(daily_sentiment) >= 2:
        latest = daily_sentiment.iloc[-1]
        previous = daily_sentiment.iloc[-2]
        delta = latest - previous
    else:
        latest = previous = delta = 0

    # direction
    if delta > 0.05:
        change_direction = "increased"
    elif delta < -0.05:
        change_direction = "decreased"
    else:
        change_direction = "stable"

    # -------------------------
    # 🔥 TOP CONTRIBUTING BRANDS
    # -------------------------

    brand_impact = (
        df.groupby("brand")["weighted_sentiment"]
        .mean()
        .sort_values(ascending=(change_direction == "decreased"))
    )

    top_brands = brand_impact.head(3).reset_index()

    top_contributing_brands = top_brands.to_dict(orient="records")

    # -------------------------
    # Extract recent articles
    # -------------------------
    recent_df = df.sort_values("date", ascending=False).head(15)

    recent_articles = recent_df[[
        "title", "brand", "topic", "weighted_sentiment"
    ]].to_dict(orient="records")

    # -------------------------
    # Build structured RAG query
    # -------------------------
    rag_query = f"""
    Market sentiment has {change_direction} recently (delta={round(float(delta), 4)}).

    Here are recent ecommerce news articles:
    {recent_articles}

    TASK:
    1. Identify the TOP 3 reasons for this sentiment change
    2. Each reason should be concise (1 line)
    3. Only use the given articles
    4. Do NOT hallucinate

    Return format:
    - reason 1
    - reason 2
    - reason 3
    """

    # -------------------------
    # Call RAG safely
    # -------------------------
    try:
        from src.rag.rag_engine import generate_rag_response

        rag_output, _ = generate_rag_response(rag_query)

        # split into top reasons
        if isinstance(rag_output, str):
            top_reasons = [
                r.strip("- ").strip()
                for r in rag_output.split("\n")
                if r.strip()
            ][:3]
        else:
            top_reasons = []

        explanation = rag_output if isinstance(rag_output, str) else ""

    except Exception as e:
        print("⚠️ RAG sentiment explanation failed:", e)
        top_reasons = []
        explanation = "Explanation unavailable due to API limits."

    # -------------------------
    # Final sentiment change object
    # -------------------------
    sentiment_change = {
        "direction": change_direction,
        "delta": round(float(delta), 4),
        "top_reasons": top_reasons,
        "explanation": explanation
    }
    # =========================================================
    # 🔥 NEW: TOP POSITIVE / NEGATIVE ARTICLES (GLOBAL)
    # =========================================================

    top_positive_articles = (
        df.sort_values("weighted_sentiment", ascending=False)
        .head(5)[["title", "url", "brand", "topic", "weighted_sentiment"]]
        .to_dict(orient="records")
    )

    top_negative_articles = (
        df.sort_values("weighted_sentiment", ascending=True)
        .head(5)[["title", "url", "brand", "topic", "weighted_sentiment"]]
        .to_dict(orient="records")
    )

    # =========================================================
    # 🔥 NEW: BRAND DRIVERS
    # =========================================================

    brand_drivers = {}

    for brand in df["brand"].unique():

        df_b = df[df["brand"] == brand]

        pos = (
            df_b.sort_values("weighted_sentiment", ascending=False)
            .head(2)[["title", "url", "topic", "weighted_sentiment"]]
            .to_dict(orient="records")
        )

        neg = (
            df_b.sort_values("weighted_sentiment", ascending=True)
            .head(2)[["title", "url", "topic", "weighted_sentiment"]]
            .to_dict(orient="records")
        )

        brand_drivers[brand] = {
            "positive_drivers": pos,
            "negative_drivers": neg
        }

    # =========================================================
    # 🔥 NEW: TOPIC DRIVERS
    # =========================================================

    topic_drivers = {}

    for topic in df["topic"].unique():

        df_t = df[df["topic"] == topic]

        pos = (
            df_t.sort_values("weighted_sentiment", ascending=False)
            .head(2)[["title", "url", "brand", "weighted_sentiment"]]
            .to_dict(orient="records")
        )

        neg = (
            df_t.sort_values("weighted_sentiment", ascending=True)
            .head(2)[["title", "url", "brand", "weighted_sentiment"]]
            .to_dict(orient="records")
        )

        topic_drivers[topic] = {
            "positive_drivers": pos,
            "negative_drivers": neg
        }

    # -------------------------
    # Final Output (EXTENDED)
    # -------------------------

    explainability_output = {
        "topic_sentiment_matrix": topic_matrix,
        "topic_explainability": topic_examples,

        # 🔥 NEW ADDITIONS
        "top_positive_articles": top_positive_articles,
        "top_negative_articles": top_negative_articles,
        "brand_drivers": brand_drivers,
        "topic_drivers": topic_drivers,
        "top_contributing_brands": top_contributing_brands,
        "sentiment_change": sentiment_change
    }

    return explainability_output