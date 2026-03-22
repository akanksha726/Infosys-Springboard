import os
from dotenv import load_dotenv
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.rag.build_vector_store import build_vector_store
from groq import Groq
from config import ECOMMERCE_BRANDS
from src.topic_modeling.topic_extractor_llm import TOPIC_KEYWORDS
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VECTOR_PATH = os.path.join(BASE_DIR, "vector_store", "news_index")

USE_LLM = True  # 🔥 control switch
ALL_TOPICS = list(TOPIC_KEYWORDS.keys())
# -----------------------------
# Load embedding + vector store
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_file = os.path.join(VECTOR_PATH, "index.faiss")

if not os.path.exists(index_file):
    print("⚠️ Vector store not found. Building now...")
    build_vector_store()

vector_store = FAISS.load_local(
    VECTOR_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # 🔥 improved context
)

# -----------------------------
# LLM client
# -----------------------------
def get_llm_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Detect query type
# -----------------------------
def detect_query_type(query):

    q = query.lower()

    if any(w in q for w in ["trend", "market", "overall"]):
        return "trend"

    elif any(w in q for w in ["risk", "alert", "warning"]):
        return "risk"

    for topic in ALL_TOPICS:

        if topic in q:
            return "topic"

    if "topic" in q:

        return "topic"

    elif any(w in q for w in ECOMMERCE_BRANDS):
        return "brand"

    return "general"


# -----------------------------
# Main RAG function
# -----------------------------
def generate_rag_response(query):

    query_type = detect_query_type(query)
    if query_type == "brand" and len(query.split()) <= 3:
        query = f"How is the brand {query} performing?"
    detected_topic = None
    for topic in ALL_TOPICS:
        if topic in query.lower():
            detected_topic = topic
            break
    client = get_llm_client()

    enhanced_query = f"""
    Indian ecommerce trends, brands like flipkart, amazon, meesho, nykaa, ajio.
    Focus on sentiment, logistics, funding, discounts.
    Question: {query}
    """

    # -----------------------------
    # Retrieve
    # -----------------------------
    docs = retriever.invoke(enhanced_query)
    # topic-focused filtering
    if detected_topic:
        docs = [d for d in docs if detected_topic in d.page_content.lower()]

    if not docs:
        return "No relevant information found.", []

    # -----------------------------
    # Filter noise
    # -----------------------------
    clean_docs = []

    for doc in docs:
        text = doc.page_content.lower()

        if any(w in text for w in ["crypto", "bitcoin", "war", "iran", "stock", "sensex"]):
            continue

        clean_docs.append(doc)

    docs = clean_docs

    if not docs:
        return "No relevant ecommerce information found.", []

    # -----------------------------
    # Sources
    # -----------------------------
    sources = [
        {"source_id": i + 1, "content": d.page_content[:200]}
        for i, d in enumerate(docs)
    ]

    # -----------------------------
    # Context
    # -----------------------------
    context = "\n\n".join([
        f"[Source {i+1}]\n{d.page_content[:400]}"
        for i, d in enumerate(docs)
    ])

    if not USE_LLM:
        return context, sources

    # -----------------------------
    # 🔥 ADAPTIVE PROMPT (CORE FIX)
    # -----------------------------
    prompt = f"""
You are an ecommerce market intelligence assistant.

QUERY TYPE: {query_type}

STRICT RULES:
- ONLY use given context
- NO external knowledge
- NO hallucination
- If unsure → say "Not explicitly mentioned"
- Give CLEAR, SYNTHESIZED answers (not raw summaries)

-----------------------
CONTEXT:
{context}
-----------------------

QUESTION:
{query}

-----------------------

INSTRUCTIONS:

IF trend:
- Identify ONE dominant trend
- Explain WHY it is happening
- Support with evidence
- Do NOT list multiple trends.
- If multiple signals exist, combine them into one unified trend.

IF brand:
- Focus ONLY on that brand
- Explain performance and sentiment
- ONLY describe what is explicitly mentioned
- DO NOT infer or assume outcomes
- If no clear evidence → say "Not explicitly mentioned"

IF risk:
- Identify risks and causes

Topics available:
discounts, logistics, competition, funding, customer_complaints,
regulation, expansion, partnership, technology

If query refers to a topic:
- Focus ONLY on that topic
- Do NOT mix unrelated topics
IF topic:
- Explain topic trends and impact

IF general:
- Provide concise structured summary

Focus ONLY on ecommerce-related implications, not general technology or macro trends.
-----------------------

OUTPUT:
Clear, structured, concise answer.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 🔥 cheaper + stable
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

    except Exception as e:
        print("⚠️ LLM error:", e)
        answer = "LLM response skipped due to rate limit."

    return answer, sources