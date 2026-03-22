import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.rag.rag_engine import generate_rag_response


def generate_market_report(base_dir):

    signal_path = os.path.join(
        base_dir,
        "data",
        "processed",
        "final_market_signal.json"
    )

    output_path = os.path.join(
        base_dir,
        "data",
        "processed",
        "market_report.txt"
    )

    with open(signal_path, "r") as f:
        data = json.load(f)

    market = data["current_market_state"]["market_signals"]

    # 🔮 Generate AI insight using RAG
    print("Generating RAG-based market insight...")

    rag_query = "Summarize the current consumer sentiment and trends in the Indian e-commerce market."

    rag_insight, rag_sources = generate_rag_response(rag_query)

    report = f"""
Ecommerce Market Intelligence Report
------------------------------------

Market Overview
The ecommerce market currently shows a {market['market_direction'].lower()} sentiment trend
with a slope of {market['market_slope']}.

Current sentiment stands at {market['current_sentiment']}.

Brand Dynamics
{market['top_positive_brand']} shows the strongest positive momentum,
while {market['top_negative_brand']} is experiencing the sharpest decline.

{market['most_volatile_brand']} currently has the highest volatility.

Narrative Trends
Top topics today include: {", ".join(market["top_topics"])}.

The most positive topic is {market["most_positive_topic"]},
while the most negative topic is {market["most_negative_topic"]}.

Emerging Narratives
Fastest rising topic: {market["fastest_rising_topic"]}

Fastest declining topic: {market["fastest_declining_topic"]}


AI Market Insight (RAG Analysis)
--------------------------------
{rag_insight}

Risk Signals
------------
{data["current_market_state"]["rag_market_risk"]}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nMarket report saved:")
    print(output_path)

    return report