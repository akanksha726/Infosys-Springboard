from src.ingestion.news_ingestion import fetch_news_for_brands, save_news_to_csv
from src.ingestion.news_ingestion import should_fetch_news
from src.preprocessing.text_preprocessing import preprocess_news_data
from src.sentiment.finbert_analyzer import run_finbert
from src.analytics.daily_sentiment_index import run_daily_sentiment_index
from src.analytics.brand_daily_index import run_brand_daily_index
from src.analytics.brand_insights import generate_brand_insights
from src.models.market_forecast import (
    forecast_market_sentiment,
    forecast_brand_sentiment,
)
from src.intelligence.market_intelligence import run_market_intelligence
from src.intelligence.event_signal_engine import run_event_signals
from src.intelligence.market_driver_detector import detect_market_drivers
from src.narrative.explainability_engine import run_explainability_engine
from src.topic_modeling.topic_extractor_llm import run_topic_extraction
from src.utils.build_master_dataset import build_master_dataset
from src.consumer.consumer_insights import generate_consumer_insights
from src.visualization.trend_visualization import run_all_trend_visuals
from src.analytics.topic_sentiment_matrix import run_topic_sentiment_matrix
from src.analytics.topic_momentum_tracker import run_topic_momentum_tracker
from src.intelligence.narrative_intelligence import run_narrative_intelligence
from src.narrative.narrative_summary import generate_market_report
from src.rag.build_vector_store import build_vector_store
from src.rag.rag_engine import generate_rag_response
from src.rag.rag_engine import generate_market_risk_signal
from config import ECOMMERCE_BRANDS

import pandas as pd
import json
from datetime import datetime
import os

def main():

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # 1️⃣ Ingestion
    print("Running ingestion...")
    brands = ECOMMERCE_BRANDS
    news_file = os.path.join(BASE_DIR, "data", "raw", "news_data.csv")
    news_updated = False
    if should_fetch_news(news_file, hours=6):
        print("Fetching fresh news...")
        news_data = fetch_news_for_brands(brands)
        save_news_to_csv(news_data)
        news_updated = True
    else:
        print("Using cached news data (no API call)")
     # 2️⃣ Preprocessing
    print("Running preprocessing...")
    preprocess_news_data()

    # -------------------------
    # 3️⃣ Sentiment Analysis
    # -------------------------

    if news_updated:
        print("Running FinBERT (new data)...")
        run_finbert()
    else:
        print("Skipping FinBERT (no new data)")

    # 4️⃣ Topic Modeling
    # -------------------------
    # 4️⃣ Topic Modeling
    # -------------------------

    if news_updated:
        print("Running topic extraction (new data)...")
        run_topic_extraction()
    else:
        print("Skipping topic extraction (no new data)")

    # 5️⃣ Build Master Dataset
    print("Building master dataset...")
    build_master_dataset()

    print("Generating trend graphs...")
    run_all_trend_visuals()

    # 5️⃣.1️⃣ Build RAG Vector Store
    print("Building vector store for RAG...")
    build_vector_store()

    # 6️⃣ Trend Indexes
    print("Generating daily sentiment index...")
    run_daily_sentiment_index()

    print("Generating brand daily sentiment index...")
    run_brand_daily_index()

    # 7️⃣ Market Intelligence
    print("Running market intelligence...")
    market_output, market_summary = run_market_intelligence()

    print("Running event signal engine...")
    event_output, event_summary = run_event_signals()

    # 8️⃣ Topic Analytics
    print("Running topic sentiment matrix...")
    topic_matrix = run_topic_sentiment_matrix()

    print("Running topic momentum tracker...")
    topic_momentum = run_topic_momentum_tracker()

    # 🔥 NEW: Driver Detection
    print("Detecting market drivers...")
    market_drivers = detect_market_drivers()

    # 9️⃣ Narrative Intelligence
    print("Running narrative intelligence...")
    narrative_output = run_narrative_intelligence()

    print("Generating AI market risk signals...")
    market_risk_signal = generate_market_risk_signal()

    # 🔮 AI Market Insight (RAG)
    print("Generating AI market insight...")

    rag_insight = generate_rag_response(
        "What are the major current trends in the Indian e-commerce market?"
    )

    print("\nAI Market Insight:")
    print(rag_insight)

    # 🔟 Forecast Engine
    print("Running market forecast...")
    market_forecast = forecast_market_sentiment()

    print("Running brand forecast...")
    brand_forecast = forecast_brand_sentiment()

    # 1️⃣1️⃣ Explainability Engine
    print("Running explainability engine...")
    explainability_output = run_explainability_engine(BASE_DIR)

    # 1️⃣2️⃣ Final Output Object
    final_output = {
        "timestamp": str(datetime.now()),

        "current_market_state": {
            "market_signals": market_output,
            "event_signals": event_output,
            "topic_sentiment_matrix": topic_matrix,
            "topic_momentum": topic_momentum,
            "market_drivers": market_drivers,
            "narrative_intelligence": narrative_output,
            "rag_market_insight": rag_insight,
            "rag_market_risk": market_risk_signal
        },

        "forecasting": {
            "market_forecast": market_forecast,
            "brand_forecast": brand_forecast
        },

        "explainability": explainability_output
    }

    output_path = os.path.join(
        BASE_DIR,
        "data/processed/final_market_signal.json"
    )

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4, default=str)

    print("Final market signal saved.")

    # 1️⃣3️⃣ AI Narrative Report
    print("Generating AI market report...")
    generate_market_report(BASE_DIR)

    brand_insights = generate_brand_insights(BASE_DIR)
    consumer_insights = generate_consumer_insights(BASE_DIR)
    # ---------------------------------------
    # Build dashboard-friendly output
    # ---------------------------------------

    dashboard_output = {

        "timestamp": str(datetime.now()),

        "market_overview": {
            "trend_direction": market_forecast["trend_direction"],
            "trend_slope": market_forecast["trend_slope"],
            "current_sentiment": market_output["current_sentiment"],
            "final_trend_score": market_output.get("current_sentiment", 0),
            "volatility": market_forecast["volatility"]
        },

        "brand_insights": {
            **brand_insights,
            "top_positive_brand": market_output["top_positive_brand"],
            "top_negative_brand": market_output["top_negative_brand"],
            "most_volatile_brand": market_output["most_volatile_brand"]
        },
        "consumer_insights": consumer_insights,

        "topic_insights": {
            "top_topics": market_output["top_topics"],
            "fastest_rising_topic": market_output["fastest_rising_topic"],
            "fastest_declining_topic": market_output["fastest_declining_topic"]
        },

        "forecast": {
            "7_day": market_forecast["forecasts"]["7_day_forecast"],
            "30_day": market_forecast["forecasts"]["30_day_forecast"],
            "90_day": market_forecast["forecasts"]["90_day_forecast"]
        },

        "ai_insight": market_output.get("rag_market_explanation", ""),

        "risk_signals": final_output["current_market_state"].get("rag_market_risk", "")
    }

    dashboard_path = os.path.join(
        BASE_DIR,
        "data/output/market_dashboard_data.json"
    )

    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    with open(dashboard_path, "w") as f:
        json.dump(dashboard_output, f, indent=4)

    print("Dashboard data exported")


if __name__ == "__main__":
    main()
