import os
import sys
import pandas as pd
import json

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src.consumer.consumer_insights import generate_consumer_insights

def verify():
    print(f"Project Base Directory: {BASE_DIR}")
    
    # Check if reviews_dataset.csv exists
    reviews_path = os.path.join(BASE_DIR, "data", "processed", "reviews_dataset.csv")
    if os.path.exists(reviews_path):
        print(f"✅ Found reviews_dataset.csv at {reviews_path}")
    else:
        print(f"❌ reviews_dataset.csv NOT found at {reviews_path}")
        return

    # Run the insights generation
    try:
        print("Starting insights generation...")
        insights = generate_consumer_insights(BASE_DIR)
        
        print("\n--- Insights Result ---")
        print(f"Sentiment Distribution: {insights['sentiment_distribution']}")
        print(f"Total Brands in Brand Sentiment: {len(insights['brand_sentiment'])}")
        print(f"AI Insight Snippet: {insights['consumer_ai_insight'][:100]}...")
        
        # Check if consumer_sentiment.csv was created
        consumer_path = os.path.join(BASE_DIR, "data", "processed", "consumer_sentiment.csv")
        if os.path.exists(consumer_path):
            print(f"✅ Created consumer_sentiment.csv at {consumer_path}")
            df = pd.read_csv(consumer_path)
            print("\nPreview of consumer_sentiment.csv:")
            print(df.head())
        else:
            print(f"❌ consumer_sentiment.csv NOT created at {consumer_path}")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify()
