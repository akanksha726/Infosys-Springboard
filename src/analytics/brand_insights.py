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
    top_brands_df = (
        master_df.groupby("brand")["final_trend_score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    top_brands = top_brands_df.to_dict(orient="records")

    # -------------------------
    # TREND DIRECTION
    # -------------------------
    brand_direction = []

    for brand in master_df["brand"].unique():
        temp = master_df[master_df["brand"] == brand].sort_values("date")

        if len(temp) < 2:
            continue

        velocity = temp["final_trend_score"].diff().iloc[-1]

        if velocity > 0:
            direction = "Rising"
        elif velocity < 0:
            direction = "Falling"
        else:
            direction = "Stable"

        brand_direction.append({
            "brand": brand,
            "direction": direction
        })

    return {
        "top_brands": top_brands,
        "brand_direction": brand_direction
    }