import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
# Ecommerce brands to track
ECOMMERCE_BRANDS = [
    "Flipkart",
    "Amazon India",
    "Myntra",
    "Snapdeal",
    "Meesho",
    "Ajio",
    "BigBasket",
    "Nykaa",
    "Reliance Digital",
    "Tata Cliq",
    "Zepto",
    "Blinkit",
    "Zomato",
    "Swiggy",
    "FirstCry",
    "Lenskart",
    "Tata 1mg",
    "Pepperfry",
    "Pharmeasy",
    "Zivame",
    "Delhivery",
    "Shiprocket",
    "H&M India",
    "Zara India",
    "Urban Company"
]

# Model settings
FINBERT_MODEL = "ProsusAI/finbert"
TOPIC_MODEL = "llama-3.3-70b-versatile"

# Pipeline settings
TOPIC_BATCH_SIZE = 10
FORECAST_DAYS = 7