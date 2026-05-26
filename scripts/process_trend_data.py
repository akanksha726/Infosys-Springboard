import pandas as pd
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "trend_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "trend_data_cleaned.csv")


# ---------------------------
# MAIN PROCESS FUNCTION
# ---------------------------
def process_trend_data():

    if not os.path.exists(INPUT_PATH):
        print("❌ Trend data not found, skipping processing")
        return

    print("Processing trend data...")

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    df = pd.read_csv(INPUT_PATH)

    # ---------------------------
    # CLEANING
    # ---------------------------

    # ensure numeric
    df['trend_score'] = pd.to_numeric(df['trend_score'], errors='coerce')

    # fill missing
    df['trend_score'] = df['trend_score'].fillna(0)

    # ---------------------------
    # DATE PARSING (ROBUST)
    # ---------------------------
    
    # First, try to parse everything directly as a single date
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')

    # If any dates failed to parse (NaT), they might be intervals like "Jan 1 – Jan 7"
    mask_nat = df['date_dt'].isna()
    if mask_nat.any():
        print(f"⚠️ Found {mask_nat.sum()} entries with complex date formats. Applying split logic...")
        
        # Split logic for "Jan 1 – Jan 7"
        split_data = df.loc[mask_nat, 'date'].str.split(r'[–-]', expand=True)
        
        if split_data.shape[1] >= 2:
            start_part = split_data[0].str.strip()
            end_part = split_data[1].str.strip()
            
            # Extract year from end_part or assume current year
            year = end_part.str.extract(r'(\d{4})')[0].fillna(str(pd.Timestamp.now().year))
            
            # Extract day for end date
            end_day = end_part.str.replace(",", "").str.extract(r'(\d{1,2})')[0]
            
            # Month from start_part
            month = start_part.str.split().str[0]
            
            # Construct start/end datetimes
            s_dt = pd.to_datetime(start_part + " " + year, format="%b %d %Y", errors='coerce')
            e_dt = pd.to_datetime(month + " " + end_day + " " + year, format="%b %d %Y", errors='coerce')
            
            # Fallback for end_date if parsing failed
            e_dt = e_dt.fillna(s_dt)
            
            # Use midpoint
            df.loc[mask_nat, 'date_dt'] = s_dt + (e_dt - s_dt) / 2
        else:
            print("❌ Failed to parse complex dates even with split logic.")

    # Drop rows that still couldn't be parsed
    df = df.dropna(subset=['date_dt'])
    df['date'] = df['date_dt']
    df = df.drop(columns=['date_dt'])

    # ---------------------------
    # AGGREGATE DUPLICATES
    # ---------------------------
    df = df.groupby(['date', 'brand'], as_index=False)['trend_score'].mean()

    # ---------------------------
    # RESAMPLING
    # ---------------------------
    # 1. Pivot to get dates as rows and brands as columns
    print("Resampling to daily frequency...")
    try:
        df_pivot = df.pivot(index='date', columns='brand', values='trend_score')
        
        # 2. Resample pivot table to daily frequency and forward fill
        df_resampled = df_pivot.resample('D').ffill()
        
        # 3. Melt back to long format
        df = df_resampled.reset_index().melt(id_vars='date', var_name='brand', value_name='trend_score')
    except Exception as e:
        print(f"⚠️ Resampling issues: {e}. Keeping original aggregate.")
        # If pivot fails (e.g. empty or duplicate issues), we stay with aggregated df

    # ---------------------------
    # SMOOTHING
    # ---------------------------
    df = df.sort_values(["brand", "date"])

    df["trend_score"] = df.groupby("brand")["trend_score"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # ---------------------------
    # NORMALIZATION
    # ---------------------------
    max_val = df["trend_score"].max()
    if max_val > 0:
        df["trend_score"] = df["trend_score"] / max_val

    # ---------------------------
    # SAVE CLEAN DATA
    # ---------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ Cleaned trend data saved!")
    print(df.head())


# ---------------------------
# RUN STANDALONE
# ---------------------------
if __name__ == "__main__":
    process_trend_data()