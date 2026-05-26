import os
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "news_master_dataset.csv")

VECTOR_PATH = os.path.join(BASE_DIR, "vector_store")


def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Master dataset not found")

    df = pd.read_csv(DATA_PATH)

    return df


def build_documents(df):

    documents = []

    # ✅ Keep topics consistent with your topic_modeling file
    VALID_TOPICS = [
        "technology",
        "logistics",
        "discounts",
        "funding",
        "competition",
        "expansion",
        "partnership",
        "customer_complaints"
    ]

    # ❌ Remove noisy macro/finance keywords
    BLOCK_KEYWORDS = [
        "crypto", "bitcoin", "sensex", "nifty",
        "stock", "ipo", "war", "iran"
    ]

    for _, row in df.iterrows():

        # -------------------------
        # Basic safety checks
        # -------------------------
        if pd.isna(row.get("brand")) or pd.isna(row.get("combined_text")):
            continue

        topic = str(row.get("topic", "other")).lower()
        text = str(row.get("combined_text", ""))

        text_lower = text.lower()

        # -------------------------
        # Filter irrelevant topics
        # -------------------------
        if topic not in VALID_TOPICS:
            continue

        # -------------------------
        # Remove noisy finance/news
        # -------------------------
        if any(word in text_lower for word in BLOCK_KEYWORDS):
            continue

        # -------------------------
        # Build document
        # -------------------------
        doc_text = f"""
Brand: {row['brand']}
Topic: {topic}
Sentiment: {row.get('finbert_label', 'neutral')}
News: {text[:1000]}
"""

        documents.append(
            Document(
                page_content=doc_text,
                metadata={
                    "brand": str(row["brand"]).lower(),
                    "topic": topic,
                    "sentiment": str(row.get("finbert_label", "neutral")).lower(),
                    "url": str(row.get("url", ""))
                }
            )
        )

    print(f"✅ Total documents after filtering: {len(documents)}")

    if len(documents) == 0:
        raise ValueError("❌ No valid documents after filtering. Check filters.")

    return documents


def build_vector_store():

    print("Loading dataset...")

    df = load_dataset()

    print("Creating filtered documents...")

    documents = build_documents(df)

    print("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")

    vector_store = FAISS.from_documents(documents, embedding_model)

    os.makedirs(VECTOR_PATH, exist_ok=True)

    vector_store.save_local(os.path.join(VECTOR_PATH, "news_index"))

    print("✅ Vector store created successfully")


if __name__ == "__main__":
    build_vector_store()