import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_trend_score():

    print("Generating Trend Score Graph...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Aggregate per day per brand
    df_grouped = df.groupby(["date", "brand"], as_index=False)["final_trend_score"].mean()

    brands = df_grouped["brand"].unique()

    plt.figure()

    for brand in brands:
        temp = df_grouped[df_grouped["brand"] == brand]
        plt.plot(temp["date"], temp["final_trend_score"], label=brand)

    plt.title("Trend Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Trend Score")
    plt.legend()
    plt.xticks(rotation=45)

    save_path = os.path.join(OUTPUT_DIR, "trend_score.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)


def plot_trend_velocity():

    print("Generating Trend Velocity Graph...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["brand", "date"])

    df["trend_velocity"] = df.groupby("brand")["final_trend_score"].diff().rolling(2).mean()

    df_grouped = df.groupby(["date", "brand"])["trend_velocity"].mean().reset_index()

    brands = df_grouped["brand"].unique()

    plt.figure()

    for brand in brands:
        temp = df_grouped[df_grouped["brand"] == brand]
        plt.plot(temp["date"], temp["trend_velocity"], label=brand)

    plt.title("Trend Velocity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Trend Velocity")
    plt.legend()
    plt.xticks(rotation=45)

    save_path = os.path.join(OUTPUT_DIR, "trend_velocity.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)


def plot_top_brands():

    print("Generating Top Brands Trend Comparison...")

    df = pd.read_csv(INPUT_PATH)

    avg_trends = df.groupby("brand")["final_trend_score"].mean().sort_values(ascending=False)

    plt.figure()

    avg_trends.plot(kind="bar")

    plt.title("Average Trend Score by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Trend Score")

    save_path = os.path.join(OUTPUT_DIR, "top_brands_trend.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)


def run_all_trend_visuals():
    plot_trend_score()
    plot_trend_velocity()
    plot_top_brands()


if __name__ == "__main__":
    run_all_trend_visuals()