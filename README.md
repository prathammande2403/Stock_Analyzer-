# 📈 Stock Analyzer (RAG + LLM)

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for stock analysis.  
It fetches **real-time stock data** and **financial news**, embeds them into a vector database, and queries an **LLM (Groq)** to provide **BUY / SELL / HOLD** recommendations with reasoning.

---

## 🚀 Features
- 📊 **Stock Data** from [Alpha Vantage](https://www.alphavantage.co/) and [Finnhub](https://finnhub.io/).  
- 📰 **News Data** from [NewsAPI](https://newsapi.org/).  
- 🔎 Embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
- 🗄️ Vector database with **FAISS** (saved locally).  
- 🤖 LLM-powered analysis via **Groq API**.  
- CLI support for ingestion & analysis.  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/prathammande2403/Stock_Analyzer-.git
cd Stock_Analyzer-
