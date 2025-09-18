#!/usr/bin/env python3
"""
stock_rag.py

RAG pipeline:
- Fetch Alpha Vantage + NewsAPI
- Build embeddings and FAISS vector store
- Retrieve context and call LLM (Groq) for BUY/SELL/HOLD decision
"""

import os
import json
import time
from typing import List, Dict

import httpx
from dotenv import load_dotenv

# LangChain pieces
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment
load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

BASE_URL_ALPHA = "https://www.alphavantage.co/query"
BASE_URL_NEWS = "https://newsapi.org/v2/everything"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# Local paths
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


# ------------------ Fetching ------------------ #
def fetch_alpha(symbol: str) -> dict:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "apikey": ALPHA_API_KEY
    }
    r = httpx.get(BASE_URL_ALPHA, params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()


def fetch_news(symbol: str, page_size: int = 20) -> dict:
    params = {
        "q": symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY
    }
    r = httpx.get(BASE_URL_NEWS, params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()

def fetch_finnhub(symbol: str) -> dict:
    """
    Fetch real-time stock quote and news from Finnhub.
    """
    data = {}

    # 1) Real-time quote
    quote_url = f"{BASE_URL_FINNHUB}/quote"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    r = httpx.get(quote_url, params=params, timeout=20.0)
    r.raise_for_status()
    data["quote"] = r.json()

    # 2) Recent news (last 7 days)
    import datetime
    today = datetime.datetime.now().date()
    week_ago = today - datetime.timedelta(days=7)
    news_url = f"{BASE_URL_FINNHUB}/company-news"
    params = {
        "symbol": symbol,
        "from": str(week_ago),
        "to": str(today),
        "token": FINNHUB_API_KEY
    }
    r = httpx.get(news_url, params=params, timeout=20.0)
    r.raise_for_status()
    data["news"] = r.json()

    return data



# ------------------ Preprocessing (make readable text for docs) ------------------ #
def make_documents_from_data(symbol: str, alpha_json: dict, news_json: dict) -> List[Document]:
    docs: List[Document] = []

    # 1) Convert latest candles into a readable block
    ts_key = "Time Series (5min)"
    candles_text = ""
    if ts_key in alpha_json:
        ts = alpha_json[ts_key]
        # get latest 10 entries sorted by timestamp descending
        items = sorted(ts.items(), key=lambda x: x[0], reverse=True)[:10]
        candles_text += f"Latest 10 five-minute candles for {symbol}:\n"
        for ts_time, fields in items:
            o = fields.get("1. open")
            h = fields.get("2. high")
            l = fields.get("3. low")
            c = fields.get("4. close")
            v = fields.get("5. volume")
            candles_text += f"{ts_time} | O:{o} H:{h} L:{l} C:{c} V:{v}\n"
    else:
        candles_text = "No intraday time series found in Alpha Vantage response.\n"

    docs.append(
        Document(
            page_content=candles_text,
            metadata={"source": "alpha_vantage", "symbol": symbol, "type": "candles"}
        )
    )

    # 2) Convert each news article into a document (title + description + url + publishedAt)
    articles = news_json.get("articles", [])
    for idx, a in enumerate(articles):
        title = a.get("title", "")
        desc = a.get("description", "")
        content = a.get("content", "")
        url = a.get("url", "")
        published = a.get("publishedAt", "")
        doc_text = f"Title: {title}\nPublishedAt: {published}\nDescription: {desc}\nContent: {content}\nURL: {url}"
        docs.append(
            Document(
                page_content=doc_text,
                metadata={"source": "newsapi", "symbol": symbol, "type": "news", "rank": idx, "url": url}
            )
        )

    return docs


# ------------------ Vector store utilities ------------------ #
def get_embeddings_model():
    # Using sentence-transformers mini model for speed
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_or_load_faiss(index_name: str, embeddings) -> FAISS:
    """
    Try to load a saved FAISS index from disk. If not present, return None.
    """
    index_path = os.path.join(VECTOR_DIR, f"{index_name}.faiss")
    meta_path = os.path.join(VECTOR_DIR, f"{index_name}_meta.json")
    if os.path.exists(index_path) and os.path.exists(meta_path):
        # load
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # FAISS.from_existing is not a standard helper; the simple approach is to load using FAISS.load_local
        # We'll use FAISS.load_local (langchain's FAISS wrapper)
        db = FAISS.load_local(VECTOR_DIR, embeddings)  # loads all artifacts; assumes single store
        return db
    else:
        return None


def persist_faiss(db: FAISS, index_name: str):
    # Persist using LangChain's save_local
    db.save_local(VECTOR_DIR)


# ------------------ Ingestion ------------------ #
def ingest_symbol(symbol: str, source="alpha", overwrite: bool = False):
    """
    Fetch data for symbol, create documents, chunk, embed and store in FAISS.
    source: "alpha" or "finnhub"
    """
    print(f"[ingest] fetching data for {symbol} from {source}...")

    if source == "alpha":
        alpha = fetch_alpha(symbol)
        time.sleep(1)
        news = fetch_news(symbol)
        raw_data = {"alpha_vantage": alpha, "news_api": news}
    elif source == "finnhub":
        raw_data = fetch_finnhub(symbol)
        # Use Finnhub news for documents
        news = raw_data.get("news", [])
        alpha = {"Time Series (5min)": {}}  # empty so make_documents_from_data doesn't fail
    else:
        raise ValueError("Unsupported source")

    # Save raw data
    raw_fname = os.path.join(DATA_DIR, f"{symbol}_raw.json")
    with open(raw_fname, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"[ingest] saved raw data to {raw_fname}")

    # create docs
    docs = make_documents_from_data(symbol, alpha, news)

    # rest of the code (chunking + FAISS) remains same...


    # chunk long docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs_split = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            doc_meta = dict(d.metadata)
            doc_meta["chunk_index"] = i
            docs_split.append(Document(page_content=c, metadata=doc_meta))

    print(f"[ingest] creating embeddings and vector store... (chunks: {len(docs_split)})")
    embed_model = get_embeddings_model()

    # create FAISS index (we'll create a new one per symbol name to keep things simple)
    # We'll put metadata in the Document objects; FAISS wrapper will keep it in-memory and save locally.
    index_name = f"{symbol.lower()}_faiss"
    # Build the vector store
    db = FAISS.from_documents(docs_split, embed_model)
    # persist
    persist_faiss(db, index_name)
    print(f"[ingest] vector store created & saved for {symbol} (index name: {index_name})")
    return db


# ------------------ Query / RAG analyze ------------------ #
def analyze_with_rag(symbol: str, question: str = "Based on the data, should I BUY, SELL, or HOLD this stock?"):
    """
    Retrieve relevant chunks for symbol and question, then call LLM to decide BUY/SELL/HOLD.
    """

    # Ensure vectorstore exists: try to load; if not present, do ingestion
    embed_model = get_embeddings_model()
    try:
        db = FAISS.load_local(VECTOR_DIR, embed_model)
    except Exception:
        print("[analyze] No existing vectorstore found; ingesting first...")
        db = ingest_symbol(symbol)

    # Retrieve top-k docs for the question
    k = 6
    retrieved = db.similarity_search(question, k=k)

    # Prepare context text for LLM
    context_blocks = []
    sources = []
    for i, doc in enumerate(retrieved):
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        url = meta.get("url", "")
        # include a small reference tag to cite
        sources.append(f"[{i+1}] {src} {('('+url+')') if url else ''}")
        block = f"--- CONTEXT {i+1} (source: {src}) ---\n{doc.page_content}\n"
        context_blocks.append(block)

    # Also include the latest candles explicitly (makes easier for LLM)
    # Try to read saved raw file to get latest candles quick
    raw_fname = os.path.join(DATA_DIR, f"{symbol}_raw.json")
    latest_candles_text = ""
    if os.path.exists(raw_fname):
        with open(raw_fname, "r") as f:
            raw = json.load(f)
        ts = raw.get("alpha_vantage", {}).get("Time Series (5min)", {})
        if ts:
            items = sorted(ts.items(), key=lambda x: x[0], reverse=True)[:10]
            latest_candles_text = "Latest candles:\n"
            for ts_time, v in items:
                latest_candles_text += f"{ts_time} | O:{v.get('1. open')} H:{v.get('2. high')} L:{v.get('3. low')} C:{v.get('4. close')} V:{v.get('5. volume')}\n"
    else:
        latest_candles_text = "No saved raw candle data available.\n"

    # Build prompt for LLM
    llm_context = "\n\n".join(context_blocks)
    full_prompt_text = f"""
You are a financial assistant. Use the provided contextual evidence (news snippets + candle data) to answer the user's question.

User question: {question}

Symbol: {symbol}

Latest market data:
{latest_candles_text}

Retrieved evidence (only use these facts; cite the evidence numbers in your reasoning):
{llm_context}

Instructions to the model:
- Respond with one of EXACT tokens: BUY / SELL / HOLD as the first line.
- On the following lines, give a short (2-3 sentences) reason for the choice, referencing which evidence items you used (e.g., [1], [3]).
- If the evidence is insufficient, say "HOLD — insufficient evidence" and explain briefly.
- Keep answer concise.
"""

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0.2
    )

    # Use a simple prompt template wrapper (LangChain ChatPromptTemplate works with the format used earlier)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful financial assistant that gives short, evidence-cited BUY/SELL/HOLD recommendations."),
        ("human", full_prompt_text)
    ])
    parser = StrOutputParser()

    prompt = prompt_template.format()  # full_prompt_text already contains everything
    response = llm.invoke(prompt)

    # parse content (response.content expected)
    model_output = response.content if hasattr(response, "content") else str(response)
    # Save the decision with metadata
    out_fname = os.path.join(DATA_DIR, f"{symbol}_decision.json")
    with open(out_fname, "w") as f:
        json.dump({"symbol": symbol, "question": question, "retrieved_sources": sources, "model_output": model_output}, f, indent=2)

    print("=== MODEL OUTPUT ===")
    print(model_output)
    print(f"[Saved decision to {out_fname}]")
    return model_output


# ------------------ Example CLI ------------------ #
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        symbol = sys.argv[2].upper()
        ingest_symbol(symbol)
    elif len(sys.argv) >= 3 and sys.argv[1] == "analyze":
        symbol = sys.argv[2].upper()
        question = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "Should I BUY, SELL, or HOLD this stock now?"
        analyze_with_rag(symbol, question)
    else:
        print("Usage:")
        print("  python stock_rag.py ingest AAPL        # fetch & build vector store for AAPL")
        print("  python stock_rag.py analyze AAPL       # run RAG analyze (uses existing store or ingests if missing)")
        print("  python stock_rag.py analyze AAPL \"Should I buy now?\"")
