from serpapi import GoogleSearch
import pandas as pd
import os
from datetime import datetime, timedelta
from config import ECOMMERCE_BRANDS, SERP_API_KEY

API_KEY = SERP_API_KEY
brands = ECOMMERCE_BRANDS

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "trend_data.csv")


# ---------------------------
# CONTROL FETCH FREQUENCY
# ---------------------------
def should_fetch_trends(file_path, hours=12):
    if not os.path.exists(file_path):
        return True

    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
    now = datetime.now()

    return (now - last_modified) > timedelta(hours=hours)


# ---------------------------
# SPLIT INTO BATCHES
# ---------------------------
def chunk_list(lst, size=5):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ---------------------------
# MAIN FETCH FUNCTION
# ---------------------------
def fetch_trends():
    if not API_KEY or API_KEY.strip() == "your_serpapi_key_here":
        print("⚠️ SERP_API_KEY not set. Skipping trend fetch.")
        return

    print("Fetching Google Trends data...")

    all_data = []

    # Use last 3 months of data for better results
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for batch in chunk_list(brands, 5):
        print(f"Fetching for: {batch}")

        params = {
            "engine": "google_trends",
            "q": ",".join(batch),
            "data_type": "TIMESERIES",
            "geo": "IN",
            "date": f"{start_date} {end_date}",
            "api_key": API_KEY
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            # --- Check for API-level errors ---
            if "error" in results:
                print(f"❌ SerpAPI Error: {results['error']}")
                continue

            # --- Check search metadata status ---
            meta = results.get("search_metadata", {})
            status = meta.get("status", "")
            if status and status != "Success":
                print(f"⚠️ Search status: {status}")

            interest = results.get("interest_over_time", {})

            if not interest:
                # Show top-level keys to help diagnose
                print(f"❌ No interest_over_time for batch: {batch}")
                print(f"   Response keys: {list(results.keys())}")
                continue

            timeline = interest.get("timeline_data", [])

            if not timeline:
                print(f"⚠️ Empty timeline_data for batch: {batch}")
                continue

            for entry in timeline:
                date = entry.get("date", "")

                for value in entry.get("values", []):
                    all_data.append({
                        "date": date,
                        "brand": value["query"].lower(),
                        "raw_score": value["value"]
                    })

        except Exception as e:
            print(f"❌ Error fetching batch {batch}: {e}")

    if not all_data:
        print("❌ No trend data collected — keeping existing cached data.")
        return

    df = pd.DataFrame(all_data)

    # ---------------------------
    # CLEAN RAW SCORE
    # ---------------------------
    df["raw_score"] = df["raw_score"].astype(str)
    df["raw_score"] = df["raw_score"].str.replace("<1", "0.5")
    df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce")
    df["raw_score"] = df["raw_score"].fillna(0)

    # ---------------------------
    # CREATE trend_score
    # ---------------------------
    max_score = df["raw_score"].max()
    if max_score == 0:
        df["trend_score"] = 0
    else:
        df["trend_score"] = df["raw_score"] / max_score

    # Brand-wise normalization
    df["trend_score"] = df.groupby("brand")["trend_score"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    df["date_parsed"] = df["date"].str.split("–").str[0]
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")

    # Smoothing
    df = df.sort_values(["brand", "date_parsed"])
    df["trend_score"] = df.groupby("brand")["trend_score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df = df.drop(columns=["date_parsed"])
    df = df[["date", "brand", "trend_score"]]
    df["fetched_at"] = pd.Timestamp.now()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Trend data saved successfully!")
    print(df.head())


# ---------------------------
# RUN STANDALONE
# ---------------------------
if __name__ == "__main__":
    if should_fetch_trends(OUTPUT_PATH):
        fetch_trends()
    else:
        print("Using cached trend data")