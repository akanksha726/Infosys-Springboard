import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "alerts.json")


def generate_alerts():

    if not os.path.exists(INPUT_PATH):
        print("❌ Master dataset not found")
        return

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("⚠️ Empty dataset — no alerts generated")
        return

    alerts = []

    # -----------------------------------
    # Get latest data per brand
    # -----------------------------------
    brand_df = df.groupby("brand").last().reset_index()

    # -----------------------------------
    # Compute statistics for anomaly detection
    # -----------------------------------
    mean_velocity = brand_df["trend_velocity"].mean()
    std_velocity = brand_df["trend_velocity"].std()

    current_time = pd.Timestamp.now().isoformat()

    # -----------------------------------
    # Alert generation loop
    # -----------------------------------
    for _, row in brand_df.iterrows():

        brand = row["brand"]
        velocity = row["trend_velocity"]
        sentiment = row["final_trend_score"]

        # -----------------------------
        # Z-SCORE (relative anomaly)
        # -----------------------------
        if std_velocity != 0:
            z_score = (velocity - mean_velocity) / std_velocity
        else:
            z_score = 0

        # -----------------------------
        # Weighted alert score
        # -----------------------------
        alert_score = (abs(velocity) * 0.6) + ((1 - sentiment) * 0.4)

        # -----------------------------
        # TREND SPIKE
        # -----------------------------
        if z_score > 1.5:
            alerts.append({
                "type": "TREND_SPIKE",
                "brand": brand,
                "message": f"{brand} showing unusual upward momentum 📈",
                "severity": "HIGH" if alert_score > 0.7 else "MEDIUM",
                "timestamp": current_time
            })

        # -----------------------------
        # TREND DROP
        # -----------------------------
        elif z_score < -1.5:
            alerts.append({
                "type": "TREND_DROP",
                "brand": brand,
                "message": f"{brand} showing unusual decline 📉",
                "severity": "HIGH" if alert_score > 0.7 else "MEDIUM",
                "timestamp": current_time
            })

        # -----------------------------
        # LOW SENTIMENT ALERT
        # -----------------------------
        sentiment_threshold = brand_df["final_trend_score"].mean() - brand_df["final_trend_score"].std()

        if sentiment < sentiment_threshold:
            severity = "HIGH" if sentiment < (sentiment_threshold - 0.1) else "MEDIUM"

            alerts.append({
                "type": "LOW_SENTIMENT",
                "brand": brand,
                "message": f"{brand} sentiment is low ⚠️",
                "severity": severity,
                "timestamp": current_time
            })
    # -----------------------------------
    # TOP PERFORMER ALERT
    # -----------------------------------
    top_brand = brand_df.sort_values("final_trend_score", ascending=False).iloc[0]

    alerts.append({
        "type": "TOP_PERFORMER",
        "brand": top_brand["brand"],
        "message": f"{top_brand['brand']} is leading the market 🚀",
        "severity": "INFO",
        "timestamp": current_time
    })

    # -----------------------------------
    # Remove duplicates (extra safety)
    # -----------------------------------
    unique_alerts = []
    seen = set()

    for alert in alerts:
        key = (alert["type"], alert["brand"])
        if key not in seen:
            unique_alerts.append(alert)
            seen.add(key)

    # -----------------------------------
    # Save alerts
    # -----------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(unique_alerts, f, indent=4)

    print(f"🚨 Alerts generated: {len(unique_alerts)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_alerts()