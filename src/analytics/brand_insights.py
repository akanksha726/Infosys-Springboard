import pandas as pd
import os


def generate_brand_insights(BASE_DIR):

    path = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")

    if not os.path.exists(path):
        print("⚠️ news_master_dataset.csv not found")
        return {
            "top_brands": [],
            "brand_direction": []
        }

    master_df = pd.read_csv(path)

    # -------------------------
    # TOP BRANDS
    # -------------------------
    try:
        top_brands_df = (
            master_df.groupby("brand")["final_trend_score"]
            .mean()
            .sort_values(ascending=False)
            .head(10) # showing more brands in top list
            .reset_index()
        )
        top_brands = top_brands_df.to_dict(orient="records")
    except:
        top_brands = []

    # -------------------------
    # TREND DIRECTION
    # -------------------------
    brand_direction = []

    # get all unique brands in the dataset
    all_brands = master_df["brand"].unique()

    for brand in all_brands:
        temp = master_df[master_df["brand"] == brand].sort_values("date")

        if len(temp) < 1:
            continue

        direction = "Stable" # default
        
        if len(temp) >= 2:
            velocity = temp["final_trend_score"].diff().iloc[-1]

            if velocity > 0.001:
                direction = "Rising"
            elif velocity < -0.001:
                direction = "Falling"
            else:
                direction = "Stable"
        else:
            # Only one data point - brand just entered monitor
            direction = "Stable"

        brand_direction.append({
            "brand": brand,
            "direction": direction
        })

    # Extra summaries for easier dashboard access
    top_pos_brand = "N/A"
    top_neg_brand = "N/A"
    most_volatile = "N/A"
    
    if len(top_brands) > 0:
        top_pos_brand = top_brands[0]["brand"]
        top_neg_brand = top_brands[-1]["brand"]
        
        # Calculate volatility if possible
        volatilities = master_df.groupby("brand")["final_trend_score"].std().sort_values(ascending=False)
        if not volatilities.empty:
            most_volatile = volatilities.index[0]

    return {
        "top_brands": top_brands,
        "brand_direction": brand_direction,
        "top_positive_brand": top_pos_brand,
        "top_negative_brand": top_neg_brand,
        "most_volatile_brand": most_volatile
    }