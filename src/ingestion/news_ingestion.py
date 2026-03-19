import requests
import pandas as pd
import os
from config import NEWS_API_KEY, ECOMMERCE_BRANDS
from datetime import datetime, timedelta
import time

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
def should_fetch_news(file_path, hours=6):
    if not os.path.exists(file_path):
        return True

    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
    now = datetime.now()

    return (now - last_modified) > timedelta(hours=hours)
# ------------------------------------------------
# Fetch News
# ------------------------------------------------

def fetch_news_for_brands(brands, page_size=5):

    all_news = []

    url = "https://newsapi.org/v2/everything"

    for brand in brands:

        print(f"Fetching news for {brand}...")

        params = {
            "q": brand,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size
        }

        try:

            response = requests.get(url, params=params, timeout=5)

            # AFTER request
            time.sleep(1.5)

            if response.status_code == 429:
                print(f"Rate limit hit for {brand}. Waiting 10 sec...")
                time.sleep(10)
                continue

            elif response.status_code != 200:
                print(f"API Error for {brand}: {response.status_code}")
                continue
            data = response.json()

            articles = data.get("articles", [])

            print(f"{brand}: {len(articles)} articles")

            for article in articles:

                all_news.append({

                    "brand": brand,

                    "source": article.get("source", {}).get("name"),

                    "title": article.get("title"),

                    "description": article.get("description"),

                    "content": article.get("content"),

                    "published_at": article.get("publishedAt"),

                    "fetched_at": datetime.now()

                })

        except Exception as e:

            print(f"Error fetching {brand}: {e}")

    print("\nTotal Articles fetched:", len(all_news))

    if len(all_news) == 0:
        print("Warning: No news articles were collected.")

    df = pd.DataFrame(all_news)

    return df


# ------------------------------------------------
# Save CSV
# ------------------------------------------------

def save_news_to_csv(df):

    save_dir = os.path.join(BASE_DIR, "data", "raw")

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "news_data.csv")

    if df is None or df.empty:
        print("No data to save. Skipping CSV creation.")
        return

    df.to_csv(file_path, index=False)

    print(f"\nNews data saved to: {file_path}")

    print(f"Rows saved: {len(df)}")


# ------------------------------------------------
# Run Script
# ------------------------------------------------

if __name__ == "__main__":

    print("Starting multi-brand news ingestion...\n")

    brands = ECOMMERCE_BRANDS

    df = fetch_news_for_brands(brands, page_size=10)

    if df is not None and not df.empty:

        save_news_to_csv(df)

        print("\nSample Data:\n")

        print(df.head())

    else:

        print("No data fetched from NewsAPI.")
