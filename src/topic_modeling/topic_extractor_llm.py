import os
import json
import pandas as pd
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# ------------------------------------------------
# Keyword topic detector (FAST + FREE)
# ------------------------------------------------

TOPIC_KEYWORDS = {

    "discounts": [
        "discount", "sale", "offer", "deal", "promotion"
    ],

    "logistics": [
        "delivery", "shipping", "warehouse", "supply chain"
    ],

    "competition": [
        "competition", "rival", "market share"
    ],

    "funding": [
        "funding", "investment", "raised", "capital"
    ],

    "customer_complaints": [
        "complaint", "refund", "delay", "issue"
    ],

    "regulation": [
        "government", "policy", "regulation", "tax"
    ],

    "expansion": [
        "expansion", "launch", "new store", "new market"
    ],

    "partnership": [
        "partnership", "collaboration", "alliance"
    ],

    "technology": [
        "ai", "technology", "automation", "platform"
    ]
}

# ------------------------------------------------
# Topic groups (higher-level themes)
# ------------------------------------------------

TOPIC_GROUPS = {

    "discounts": "Pricing Strategy",
    "competition": "Market Competition",
    "logistics": "Operations",
    "technology": "Innovation",
    "expansion": "Growth",
    "funding": "Finance",
    "customer_complaints": "Customer Experience",
    "regulation": "Policy",
    "partnership": "Business Strategy",
    "other": "Misc"
}

# ------------------------------------------------
# Fast keyword classifier
# ------------------------------------------------

def keyword_topic_detector(text):

    text = text.lower()

    for topic, words in TOPIC_KEYWORDS.items():

        for w in words:

            if w in text:
                return topic

    return None


# ------------------------------------------------
# LLM fallback classifier
# ------------------------------------------------

def extract_topics_llm(texts):

    numbered_text = "\n\n".join(
        [f"{i+1}. {t}" for i, t in enumerate(texts)]
    )

    prompt = f"""
You are analyzing Indian e-commerce news articles.

Classify the MAIN topic of each article.

Possible topics:
Discounts
Logistics
Competition
Funding
Customer Complaints
Regulation
Expansion
Partnership
Technology
Other

Return ONLY JSON list like:

[
{{"topic":"Expansion"}},
{{"topic":"Technology"}}
]

Articles:
{numbered_text}
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {"role": "user", "content": prompt}
            ],

            temperature=0
        )

        result = response.choices[0].message.content

        json_match = re.search(r"\[.*\]", result, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

    except Exception as e:
        print("LLM ERROR:", e)

    return [{"topic": "Other"}] * len(texts)


# ------------------------------------------------
# Normalize topic names
# ------------------------------------------------

def normalize_topic(topic):

    topic = str(topic).lower()

    topic_map = {
        "discounts": "discounts",
        "logistics": "logistics",
        "competition": "competition",
        "funding": "funding",
        "customer complaints": "customer_complaints",
        "regulation": "regulation",
        "expansion": "expansion",
        "partnership": "partnership",
        "technology": "technology",
    }

    for k in topic_map:

        if k in topic:
            return topic_map[k]

    return "other"


# ------------------------------------------------
# Main topic extraction pipeline
# ------------------------------------------------

def run_topic_extraction():

    input_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "cleaned_news.csv"
    )

    output_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "news_with_topics.csv"
    )

    df = pd.read_csv(input_path)

    texts = df["Combined_Text"].fillna("").tolist()

    topics = []

    llm_queue = []
    confidences = []
    llm_indices = []

    # ----------------------------
    # Step 1: Keyword classification
    # ----------------------------

    for i, text in enumerate(texts):

        topic = keyword_topic_detector(text)

        if topic is None:

            llm_queue.append(text)
            llm_indices.append(i)
            topics.append(None)

        else:

            topics.append(topic)
            confidences.append(1.0)

    # ----------------------------
    # Step 2: LLM fallback
    # ----------------------------

    if llm_queue:

        print(f"Sending {len(llm_queue)} articles to LLM...")

        results = extract_topics_llm(llm_queue)

        for idx, r in zip(llm_indices, results):

            topic_raw = r.get("topic", "Other")

            topics[idx] = normalize_topic(topic_raw)
            confidences.insert(idx, 0.8)

    # ----------------------------
    # Save results
    # ----------------------------

    df["topic"] = topics
    # Fix length mismatch
    min_len = min(len(df), len(confidences))

    df = df.iloc[:min_len].copy()

    df["topic_confidence"] = confidences[:min_len]

    df["topic_group"] = df["topic"].map(TOPIC_GROUPS)
    df["topic_group"] = df["topic_group"].fillna("Misc")

    df.to_csv(output_path, index=False)

    print("✅ Topic extraction completed")

    print(f"Saved to: {output_path}")
