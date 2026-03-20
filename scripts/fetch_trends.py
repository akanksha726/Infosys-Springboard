from serpapi import GoogleSearch
import pandas as pd
import os
from config import ECOMMERCE_BRANDS, SERP_API_KEY

API_KEY = SERP_API_KEY
brands = ECOMMERCE_BRANDS
OUTPUT_PATH = "../data/processed/trend_data.csv"

# ---------------------------
# Helper: split into chunks of 5
# ---------------------------
def chunk_list(lst, size=5):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

all_data = []

# ---------------------------
# FETCH IN BATCHES
# ---------------------------
for batch in chunk_list(brands, 5):
    print(f"Fetching for: {batch}")

    params = {
        "engine": "google_trends",
        "q": ",".join(batch),
        "data_type": "TIMESERIES",
        "geo": "IN",
        "api_key": API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    interest = results.get("interest_over_time", {})

    if not interest:
        print("❌ No data for batch:", batch)
        print(results)
        continue

    timeline = interest.get("timeline_data", [])

    for entry in timeline:
        date = entry["date"]

        for value in entry["values"]:
            all_data.append({
                "date": date,
                "brand": value["query"],
                "trend_score": value["value"]
            })

# ---------------------------
# SAVE
# ---------------------------
df = pd.DataFrame(all_data)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Trend data saved successfully!")
print(df.head())