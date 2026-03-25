import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from config import ECOMMERCE_BRANDS
from src.utils.data_loader import load_daily_metrics, load_brand_metrics
from src.rag.rag_engine import generate_rag_response
from src.utils.data_loader import load_master_dataset

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ------------------------------------------------
# Load intelligence signals
# ------------------------------------------------

def load_intelligence_signals():

    signals = {}

    try:
        topic_momentum = pd.read_csv(
            os.path.join(BASE_DIR, "data/processed/topic_momentum.csv")
        )
        signals["topic_momentum"] = topic_momentum
    except:
        signals["topic_momentum"] = None

    try:
        topic_sentiment = pd.read_csv(
            os.path.join(BASE_DIR, "data/processed/topic_sentiment_matrix.csv")
        )
        signals["topic_sentiment"] = topic_sentiment
    except:
        signals["topic_sentiment"] = None

    try:
        event_signals = pd.read_json(
            os.path.join(BASE_DIR, "data/processed/event_signals.json")
        )
        signals["event_signals"] = event_signals
    except:
        signals["event_signals"] = None

    return signals


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


# ------------------------------------------------
# Walk-forward validation
# ------------------------------------------------

def walk_forward_validation(df, feature_cols):

    preds = []
    actuals = []

    for i in range(6, len(df) - 1):

        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        X_train = train[feature_cols]
        y_train = train["sentiment_index"]

        X_test = test[feature_cols]
        y_test = test["sentiment_index"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test.values[0])

    return evaluate_model(actuals, preds)


# ------------------------------------------------
# Forecast Driver Generator
# ------------------------------------------------

def generate_forecast_drivers(feature_importance):

    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    drivers = []

    for feature, score in sorted_features[:3]:

        readable = feature.replace("_", " ")

        drivers.append({
            "driver": readable,
            "importance": round(float(score), 3)
        })

    return drivers


# ------------------------------------------------
# Market Forecast
# ------------------------------------------------

def forecast_market_sentiment():

    df = load_daily_metrics()

    output_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "market_forecast_history.csv"
    )
    master_df = load_master_dataset()

    # convert dates
    df["date"] = pd.to_datetime(df["date"])
    master_df["date"] = pd.to_datetime(master_df["date"])

    # aggregate final trend per day
    trend_daily = (
        master_df.groupby("date")["final_trend_score"]
        .mean()
        .reset_index()
    )

    # merge into df
    df = pd.merge(
        df,
        trend_daily,
        on="date",
        how="left"
    )
    # -------------------------
    # 🔥 ADD INTELLIGENCE FEATURES (NEW)
    # -------------------------

    # narrative risk (proxy using topics)
    risk_topics = ["logistics", "regulation"]

    existing_cols = [col for col in risk_topics if col in df.columns]

    df["narrative_risk_score"] = (
        df[existing_cols].sum(axis=1)
        if existing_cols else 0
    )

    # market shock
    df["market_shock"] = (
            df["final_trend_score"] -
            df["final_trend_score"].rolling(5).mean()
    ).fillna(0)

    # fill missing
    df["final_trend_score"] = df["final_trend_score"].fillna(0)

    if df.empty or len(df) < 10:
        return {"error": "Not enough data for forecasting"}

    # -------------------------
    # ADD TREND FEATURES SAFETY
    # -------------------------
    for col in ["trend_score", "trend_velocity", "trend_sentiment_signal"]:
        if col not in df.columns:
            df[col] = 0

    df = df.sort_values("date")

    df["time_index"] = range(len(df))

    df["sentiment_lag_1"] = df["sentiment_index"].shift(1)
    df["sentiment_lag_2"] = df["sentiment_index"].shift(2)

    df["rolling_mean_3"] = df["sentiment_index"].rolling(3).mean()

    df["sentiment_momentum"] = df["sentiment_index"].diff()
    df["sentiment_acceleration"] = df["sentiment_momentum"].diff()

    df["sin_week"] = np.sin(2 * np.pi * df["time_index"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["time_index"] / 7)

    topic_cols = [
        "customer_complaints",
        "discounts",
        "expansion",
        "funding",
        "logistics",
        "other",
        "partnership",
        "regulations",
        "technology"
    ]

    for col in topic_cols:
        if col not in df.columns:
            df[col] = 0

    momentum = df[topic_cols].diff()
    df["topic_momentum_score"] = momentum.abs().sum(axis=1)

    positive_topics = [
        "expansion",
        "funding",
        "technology",
        "partnership"
    ]

    negative_topics = [
        "customer_complaints",
        "logistics",
        "regulations"
    ]

    df["avg_topic_sentiment"] = (
            df[positive_topics].sum(axis=1)
            - df[negative_topics].sum(axis=1)
    )

    df["event_intensity"] = df[topic_cols].sum(axis=1)
    df = df.fillna(0)

    df["topic_intensity_3"] = df["event_intensity"].rolling(3).mean()
    df["topic_intensity_5"] = df["event_intensity"].rolling(5).mean()

    df = df.fillna(0)

    feature_cols = [
        "time_index",
        "sentiment_lag_1",
        "sentiment_lag_2",
        "rolling_mean_3",
        "sentiment_momentum",
        "sentiment_acceleration",
        "sin_week",
        "cos_week",
        "topic_momentum_score",
        "avg_topic_sentiment",
        "event_intensity",
        "topic_intensity_3",
        "topic_intensity_5",
        "final_trend_score",
        "trend_velocity",
        "trend_sentiment_signal",
        "narrative_risk_score",
        "market_shock"
    ]

    X = df[feature_cols]
    y = df["sentiment_index"]

    evaluation = walk_forward_validation(df, feature_cols)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    feature_importance = dict(
        zip(feature_cols, model.feature_importances_)
    )

    drivers = generate_forecast_drivers(feature_importance)

    last_index = df["time_index"].iloc[-1]
    last_values = df.tail(7)["sentiment_index"].tolist()

    slope = np.polyfit(df["time_index"], df["sentiment_index"], 1)[0]

    def generate_forecast(days):

        preds = []

        current_index = last_index + 1
        values = last_values.copy()

        for i in range(days):

            lag1 = values[-1]
            lag2 = values[-2]

            rolling = np.mean(values[-3:])

            sin_week = np.sin(2 * np.pi * current_index / 7)
            cos_week = np.cos(2 * np.pi * current_index / 7)

            momentum = lag1 - lag2
            acceleration = momentum - (values[-2] - values[-3]) if len(values) >= 3 else 0

            features = pd.DataFrame([{
                "time_index": current_index,
                "sentiment_lag_1": lag1,
                "sentiment_lag_2": lag2,
                "rolling_mean_3": rolling,
                "sentiment_momentum": momentum,
                "sentiment_acceleration": acceleration,
                "sin_week": sin_week,
                "cos_week": cos_week,
                "topic_momentum_score": df["topic_momentum_score"].tail(3).mean(),
                "avg_topic_sentiment": df["avg_topic_sentiment"].tail(3).mean(),
                "event_intensity": df["event_intensity"].tail(3).mean(),
                "topic_intensity_3": df["topic_intensity_3"].tail(3).mean(),
                "topic_intensity_5": df["topic_intensity_5"].tail(3).mean(),
                "final_trend_score": df["final_trend_score"].tail(3).mean(),
                "trend_velocity": df["trend_velocity"].tail(3).mean(),
                "trend_sentiment_signal": df["trend_sentiment_signal"].tail(3).mean(),
                "narrative_risk_score": df["narrative_risk_score"].tail(3).mean(),
                "market_shock": df["market_shock"].tail(3).mean()
            }])

            pred = model.predict(features)[0]
            pred += slope * 0.2
            pred = max(-1, min(1, pred))

            preds.append(round(float(pred), 4))
            values.append(pred)

            current_index += 1

        return preds


    forecast_7 = generate_forecast(7)
    forecast_30 = generate_forecast(30)
    forecast_90 = generate_forecast(90)

    trend = "Stable"

    if slope > 0.01:
        trend = "Bullish"
    elif slope < -0.01:
        trend = "Bearish"

    volatility = float(np.std(y))
    confidence = max(0.1, 1 - volatility)

    # 🔮 RAG FORECAST EXPLANATION
    rag_query = f"""
Explain the predicted ecommerce market sentiment trend.

Trend Direction: {trend}
Market Volatility: {round(volatility,4)}
Key Forecast Drivers: {drivers}

Provide a short explanation of why the forecast may be occurring based on recent ecommerce news.
"""

    rag_forecast_explanation, rag_sources = generate_rag_response(rag_query)
    run_date = pd.Timestamp.now().date()
    # -------------------------
    # SAVE FORECAST HISTORY (NEW)
    # -------------------------

    forecast_record = pd.DataFrame([{
        "date": run_date,
        "trend_direction": trend,
        "trend_slope": round(float(slope), 4),
        "volatility": round(volatility, 4),
        "confidence": round(confidence, 3),
        "forecast_7": str(forecast_7),
        "forecast_30": str(forecast_30),
        "forecast_90": str(forecast_90)
    }])

    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)

        old_df["date"] = pd.to_datetime(old_df["date"])
        forecast_record["date"] = pd.to_datetime(forecast_record["date"])

        combined = pd.concat([old_df, forecast_record])

        # keep latest run per date
        combined = combined.drop_duplicates(
            subset=["date"],
            keep="last"
        )
    else:
        combined = forecast_record

    combined.to_csv(output_path, index=False)

    return {

        "trend_direction": trend,

        "trend_slope": round(float(slope), 4),

        "volatility": round(volatility, 4),

        "confidence_score": round(confidence, 3),

        "evaluation": evaluation,

        "feature_importance": {
            k: round(float(v), 3)
            for k, v in feature_importance.items()
        },

        "forecast_drivers": drivers,

        "rag_forecast_explanation": rag_forecast_explanation,
        "rag_forecast_sources": rag_sources,

        "forecasts": {
            "7_day_forecast": forecast_7,
            "30_day_forecast": forecast_30,
            "90_day_forecast": forecast_90
        }
    }

def forecast_brand_sentiment():

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    df = load_brand_metrics()

    output_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "brand_forecast_history.csv"
    )

    run_date = pd.Timestamp.now().date()
    forecast_records = []

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if df is None or df.empty:
        return {"brand_forecasts": []}

    # -------------------------
    # PREPROCESS
    # -------------------------
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

    # 🔥 ensure correct column name
    if "sentiment_index" not in df.columns:
        if "sentiment" in df.columns:
            df["sentiment_index"] = df["sentiment"]
        elif "score" in df.columns:
            df["sentiment_index"] = df["score"]
        else:
            raise ValueError("❌ sentiment column not found in brand metrics")

    # 🔥 ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.sort_values("date")

    results = []

    all_brands = [b.lower() for b in ECOMMERCE_BRANDS]

    # -------------------------
    # LOOP OVER BRANDS
    # -------------------------
    for brand in all_brands:

        brand_df = df[df["brand"] == brand].copy()

        # -------------------------
        # NO DATA
        # -------------------------
        if brand_df.empty:
            results.append({
                "brand": brand,
                "trend_direction": "No Data",
                "trend_slope": 0,
                "forecasts": {
                    "7_day": [],
                    "30_day": [],
                    "90_day": []
                }
            })
            continue

        brand_df = brand_df.sort_values("date")

        # -------------------------
        # MIN DATA
        # -------------------------
        if len(brand_df) < 2:
            results.append({
                "brand": brand,
                "trend_direction": "Insufficient Data",
                "trend_slope": 0,
                "forecasts": {
                    "7_day": [],
                    "30_day": [],
                    "90_day": []
                }
            })
            continue

        # -------------------------
        # FEATURE ENGINEERING
        # -------------------------
        brand_df["time_index"] = range(len(brand_df))

        # 🔥 momentum (key fix)
        brand_df["momentum"] = brand_df["sentiment_index"].diff().fillna(0)

        feature_cols = ["time_index", "momentum"]

        X = brand_df[feature_cols]
        y = brand_df["sentiment_index"]

        # -------------------------
        # MODEL
        # -------------------------
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X, y)

        # -------------------------
        # TREND DETECTION
        # -------------------------
        slope = np.polyfit(
            brand_df["time_index"],
            brand_df["sentiment_index"],
            1
        )[0]

        slope = max(-0.5, min(0.5, slope))  # clamp

        if slope > 0.01:
            direction = "Improving"
        elif slope < -0.01:
            direction = "Declining"
        else:
            direction = "Stable"

        # -------------------------
        # FORECAST FUNCTION
        # -------------------------
        def generate(days):

            future = pd.DataFrame({
                "time_index": range(len(brand_df), len(brand_df) + days)
            })

            # 🔥 dynamic momentum
            last_momentum = brand_df["momentum"].iloc[-1]
            future["momentum"] = last_momentum

            preds = model.predict(future)

            # 🔥 add trend effect (CRITICAL FIX)
            preds = [
                p + slope * i * 0.3 for i, p in enumerate(preds)
            ]

            # 🔥 clamp values
            preds = [max(-1, min(1, p)) for p in preds]

            return [round(float(p), 4) for p in preds]

        # -------------------------
        # STORE RESULT
        # -------------------------
        forecast_7 = generate(7)
        forecast_30 = generate(30)
        forecast_90 = generate(90)

        results.append({
            "brand": brand,
            "trend_direction": direction,
            "trend_slope": round(float(slope), 4),
            "forecasts": {
                "7_day": forecast_7,
                "30_day": forecast_30,
                "90_day": forecast_90
            }
        })

        # SAVE HISTORY
        forecast_records.append({
            "date": run_date,
            "brand": brand,
            "trend_direction": direction,
            "trend_slope": round(float(slope), 4),
            "forecast_7": str(forecast_7),
            "forecast_30": str(forecast_30),
            "forecast_90": str(forecast_90)
        })
        forecast_df = pd.DataFrame(forecast_records)

        if os.path.exists(output_path):
            old_df = pd.read_csv(output_path)

            old_df["date"] = pd.to_datetime(old_df["date"])
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            combined = pd.concat([old_df, forecast_df])

            combined = combined.drop_duplicates(
                subset=["date", "brand"],
                keep="last"
            )
        else:
            combined = forecast_df

        combined.to_csv(output_path, index=False)
    return {
        "brand_forecasts": results
    }