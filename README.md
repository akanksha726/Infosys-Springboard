# 🚀 AI-Powered Market Trend & Consumer Sentiment Forecaster

An end-to-end AI system that analyzes ecommerce market trends using multi-source data (news + Google Trends + consumer signals) and generates actionable business insights with explainability.
LIVE PREVIEW

https://infosys-springboard-git-c043f5-thakurakanksha837-4728s-projects.vercel.app/

---

## 📌 Overview

This project builds a **real-time market intelligence pipeline** that:

- Aggregates ecommerce news and Google Trends data
- Performs sentiment analysis using FinBERT
- Extracts topics using LLM-based modeling
- Computes trend scores, momentum, and volatility
- Generates explainable insights using RAG (Retrieval-Augmented Generation)
- Produces alerts, forecasts, and PDF reports

---
## 🔄 System Workflow

```text
News API + Google Trends
        ↓
Data Preprocessing
        ↓
Sentiment Analysis (FinBERT)
        ↓
Topic Modeling (LLM)
        ↓
Feature Engineering (trend, velocity, entropy)
        ↓
Market Intelligence Engine
        ↓
Forecasting + Signals
        ↓
RAG-based Explainability
        ↓
Reports + Alerts + Dashboard Data
```
---
## 🧠 Key Features

✔ Multi-source data ingestion (News + Google Trends)  
✔ NLP-based sentiment analysis (FinBERT)  
✔ Topic modeling using LLM  
✔ Time-series trend analysis (momentum, velocity, entropy)  
✔ Market intelligence engine (signals, risks, drivers)  
✔ RAG-based explainable insights  
✔ Forecasting module  
✔ Alert generation system  
✔ Automated PDF report generation  
✔ Dashboard-ready JSON export  

---

## 🏗️ Project Architecture

```
TEAM-REPO/
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── output/
│ └── archive/
│
├── scripts/
│ ├── fetch_trends.py
│ ├── process_trend_data.py
│ └── alerts/
│
├── src/
│ ├── ingestion/
│ ├── preprocessing/
│ ├── sentiment/
│ ├── topic_modeling/
│ ├── analytics/
│ ├── consumer/
│ ├── intelligence/
│ ├── models/
│ ├── narrative/
│ ├── rag/
│ ├── reporting/
│ ├── visualization/
│ └── utils/
│
├── vector_store/
├── config.py
├── run_market_engine.py
├── requirements.txt
├── .env.example
└── README.md
````

### Directory Notes

**src/**  
Contains the core modules of the project pipeline including ingestion, sentiment analysis, topic modeling, forecasting models, and RAG engines.

**data/**  
Stores raw, processed, and output datasets generated throughout the pipeline.

**vector_store/**  
FAISS vector database used for semantic retrieval in the RAG insight engine.

---

## Installation & Setup

### Clone Repository
```bash
git clone https://github.com/your-repo-name.git
cd AI-Ecommerce-Trend-Forecaster
````

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

Windows

```bash
.venv\Scripts\activate
```

Mac/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_groq_api_key
NEWS_API_KEY=your_newsapi_key
SERP_API_KEY=your_serpapi_key
```

These are required for:

* Groq LLM API
* NewsAPI ingestion
* SerpAPI Google Trends data
---

## Running the Pipeline


```bash
python run_market_engine.py
```
## Outputs Generated
### 📁 Processed Data
* news_master_dataset.csv
* daily_market_metrics.csv
* brand_daily_metrics.csv
* consumer_sentiment.csv
### 📈 Analytics
* Trend scores & velocity
* Topic momentum & sentiment matrix
* Market signals & risk indicators
### 📄 Reports
* market_report.txt
* market_report.pdf
### 📊 Visualizations
* trend_score.png
* trend_velocity.png
* top_brands_trend.png
### 🚨 Alerts
* alerts.json
### 📦 Dashboard Data
* market_dashboard_data.json
This runs:

1. News ingestion
2. Text preprocessing
3. FinBERT sentiment analysis
4. LLM topic extraction
5. Master dataset creation
6. Market intelligence analytics
7. Event signal detection
8. Narrative intelligence
9. Forecast generation
10. RAG market insights
11. Explainability engine
12. AI market report generation

---

## Output Files

### Structured Data

```
data/processed/final_market_signal.json
```

Contains:

* market signals
* topic insights
* event signals
* forecast predictions
* RAG explanations

### AI Generated Market Report

```
data/output/market_report.pdf
```

Includes:

* market overview
* brand dynamics
* narrative trends
* emerging risks
* AI-generated strategic insights

---

## Example Forecast Output

Trend Direction: Bearish
Trend Slope: -0.023
Volatility: 0.46
Confidence Score: 0.53

Forecast Horizons:

* 7 Day Forecast
* 30 Day Forecast
* 90 Day Forecast

---

## Technologies Used

### NLP / AI

* FinBERT
* Llama 3
* HuggingFace Transformers
* LangChain

### Machine Learning

* Random Forest Regression
* Feature Engineering
* Walk-forward validation

### Retrieval Augmented Generation

* FAISS Vector Database
* Sentence Transformers
* Groq API

### Data Processing

* Pandas
* NumPy

---

## Model Evaluation

Example evaluation metrics:

```
MAE: 0.42
RMSE: 0.47
R²: -3.25
```

Low R² occurs due to limited historical data (~14 days).

With more data (40–60 days), performance improves significantly.

---

## License

This project is developed for educational and research purposes.

---

## Contributors

Project developed as part of an AI Market Intelligence System for analyzing e-commerce sentiment and trends.

