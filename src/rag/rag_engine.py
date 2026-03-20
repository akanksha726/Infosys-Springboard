import os
from dotenv import load_dotenv
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rag.build_vector_store import build_vector_store
from groq import Groq


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

VECTOR_PATH = os.path.join(BASE_DIR, "vector_store", "news_index")


# -----------------------------
# Load embedding + vector store once
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Ensure vector store exists
# -----------------------------
index_file = os.path.join(VECTOR_PATH, "index.faiss")

if not os.path.exists(index_file):
    print("⚠️ Vector store not found. Building now...")
    build_vector_store()

# -----------------------------
# Load vector store
# -----------------------------
vector_store = FAISS.load_local(
    VECTOR_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# LLM client
# -----------------------------

def get_llm_client():

    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# RAG response
# -----------------------------

def generate_rag_response(query):

    client = get_llm_client()

    docs = retriever.invoke(query)

    sources = [doc.page_content[:200] for doc in docs]

    context = "\n\n".join([doc.page_content[:400] for doc in docs])

    prompt = f"""
You are a senior ecommerce market intelligence analyst.

Analyze the news context and answer the query using structured reasoning.

Context:
{context}

Question:
{query}

Follow this reasoning structure:

1. Key Market Signals
Identify the main signals emerging from the news.

2. Brand Impact
Explain which ecommerce brands are most affected.

3. Consumer Sentiment Drivers
Explain why sentiment is positive or negative.

4. Strategic Insight
Provide one concise strategic takeaway for market observers.

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content, sources


# -----------------------------
# Risk signal detector
# -----------------------------

def generate_market_risk_signal():

    query = """
Analyze the ecommerce news context and identify any emerging market risks.

Classify risks into the following categories:

1. Regulatory risk
2. Logistics risk
3. Competition pressure
4. Consumer complaints
5. Technology disruption

Return a short structured summary of detected risks.
"""

    insight, sources = generate_rag_response(query)

    return insight