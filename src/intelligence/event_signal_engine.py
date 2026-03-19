import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from src.utils.data_loader import load_master_dataset

def run_event_signals():

    print("Running Event Signal Engine...")

    df = load_master_dataset()

    if df.empty:
        print("Dataset empty.")
        return {}, "No event signals available."

    # ---------------------------
    # 1️⃣ Date Processing
    # ---------------------------

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.date

    df["topic"] = df["topic"].fillna("Other")
    df["finbert_label"] = df["finbert_label"].fillna("neutral")

    # ---------------------------
    # 2️⃣ Topic-Based Flags
    # ---------------------------

    # Competition signals
    df["competition_flag"] = df["topic"].isin(["Discounts", "Competition"])

    # Complaint signals
    df["complaint_flag"] = (
        (df["topic"] == "Customer Complaints")
        & (df["finbert_label"] == "negative")
    )

    # ---------------------------
    # 3️⃣ Daily Aggregation
    # ---------------------------

    daily_signals = df.groupby("date").agg(
        competition_count=("competition_flag", "sum"),
        complaint_count=("complaint_flag", "sum"),
        article_volume=("topic", "count")
    ).reset_index()

    # 7-day moving average
    daily_signals["volume_ma7"] = daily_signals["article_volume"].rolling(
        7, min_periods=1
    ).mean()

    latest = daily_signals.iloc[-1]

    # ---------------------------
    # 4️⃣ Competition Signal
    # ---------------------------

    competition_ma7 = daily_signals["competition_count"].rolling(
        7, min_periods=1
    ).mean().iloc[-1]

    competition_today = latest["competition_count"]

    if competition_ma7 == 0:
        competition_ratio = competition_today
    else:
        competition_ratio = competition_today / competition_ma7

    if competition_ratio >= 2:
        competition_level = "High"
    elif competition_ratio > 1:
        competition_level = "Moderate"
    elif competition_today == 0:
        competition_level = "Low"
    else:
        competition_level = "Stable"

    # ---------------------------
    # 5️⃣ Complaint Signal
    # ---------------------------

    complaint_ma7 = daily_signals["complaint_count"].rolling(
        7, min_periods=1
    ).mean().iloc[-1]

    complaint_today = latest["complaint_count"]

    if complaint_ma7 == 0:
        complaint_ratio = complaint_today
    else:
        complaint_ratio = complaint_today / complaint_ma7

    if complaint_ratio >= 2:
        complaint_level = "High"
    elif complaint_ratio > 1:
        complaint_level = "Moderate"
    elif complaint_today == 0:
        complaint_level = "Low"
    else:
        complaint_level = "Stable"

    # ---------------------------
    # 6️⃣ Sector Narrative Heat
    # ---------------------------

    volume_today = latest["article_volume"]
    volume_ma7 = latest["volume_ma7"]

    heat_ratio = volume_today / volume_ma7 if volume_ma7 != 0 else 0

    if heat_ratio >= 1.5:
        sector_heat = "High Narrative Heat"
    elif heat_ratio > 1.1:
        sector_heat = "Elevated Attention"
    elif heat_ratio < 0.8:
        sector_heat = "Cooling"
    else:
        sector_heat = "Stable"

    # ---------------------------
    # 7️⃣ Brand Drivers
    # ---------------------------

    competition_brands = (
        df[df["competition_flag"]]["brand"]
        .value_counts()
        .index
        .tolist()
    )

    complaint_brands = (
        df[df["complaint_flag"]]["brand"]
        .value_counts()
        .index
        .tolist()
    )

    # ---------------------------
    # 8️⃣ Narrative Summary
    # ---------------------------

    summary = f"""
    Current ecommerce market signals indicate competitive promotional activity.

    Competition intensity is {competition_level}, 
    with activity running at {round(competition_ratio,2)} times the weekly average.

    Complaint-related coverage is {complaint_level.lower()},
    suggesting limited consumer grievance narratives.

    Sector narrative heat is {sector_heat},
    with article volume at {round(heat_ratio,2)} times the baseline.

    Promotion drivers: {competition_brands if competition_brands else "None"}.
    Complaint drivers: {complaint_brands if complaint_brands else "None"}.
    """

    # ---------------------------
    # 9️⃣ Structured Output
    # ---------------------------

    output = {
        "competition_intensity": competition_level,
        "competition_ratio": round(float(competition_ratio), 2),
        "complaint_pressure": complaint_level,
        "complaint_ratio": round(float(complaint_ratio), 2),
        "sector_heat": sector_heat,
        "heat_ratio": round(float(heat_ratio), 2),
        "competition_brands": competition_brands,
        "complaint_brands": complaint_brands
    }

    print("\n--- Analytical Signals ---")
    print(output)

    print("\n--- Narrative Summary ---")
    print(summary)

    return output, summary


if __name__ == "__main__":
    output, summary = run_event_signals()
    print(summary)
    print(output)