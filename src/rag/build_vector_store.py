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

    for _, row in df.iterrows():

        text = f"""
Brand: {row['brand']}
Topic: {row['topic']}
Sentiment: {row['finbert_label']}
News: {row['combined_text'][:1000]}
"""

        documents.append(Document(page_content=text))

    return documents


def build_vector_store():

    print("Loading dataset...")

    df = load_dataset()

    print("Creating documents...")

    documents = build_documents(df)

    print("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")

    vector_store = FAISS.from_documents(documents, embedding_model)

    os.makedirs(VECTOR_PATH, exist_ok=True)

    vector_store.save_local(os.path.join(VECTOR_PATH, "news_index"))

    print("Vector store created successfully")


if __name__ == "__main__":
    build_vector_store()