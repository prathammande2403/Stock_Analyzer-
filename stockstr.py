import streamlit as st
import os
import json
from stock import fetch_alpha_daily, fetch_news, analyze_stock



st.set_page_config(page_title="Stock Analyzer RAG", layout="centered")
st.title("📈 Stock Analyzer (RAG)")

# Input: stock symbol
symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA):").upper()

if st.button("Analyze Stock") and symbol:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            # Fetch latest price from Alpha Vantage
            alpha_data = fetch_alpha_daily(symbol)
            
            # Fetch latest price from Alpha Vantage (daily)
            ts_key = "Time Series (Daily)"
            latest_price = "N/A"
            if ts_key in alpha_data:
               latest_time = sorted(alpha_data[ts_key].keys(), reverse=True)[0]
               latest_price = alpha_data[ts_key][latest_time]["4. close"]


            # Fetch news (optional, Streamlit can show top 5)
            news_data = fetch_news(symbol)
            articles = news_data.get("articles", [])[:5]

            # Run RAG pipeline
            action = analyze_stock(symbol)

        # Display results
        st.subheader(f"Latest Price for {symbol}: ${latest_price}")
        st.subheader("LLM Recommendation:")
        st.success(action)

        st.subheader("Top 5 News Headlines:")
        for i, article in enumerate(articles, 1):
            st.write(f"{i}. [{article.get('title','No title')}]({article.get('url','')})")
            
    except Exception as e:
        st.error(f" Error fetching or analyzing data: {e}")
        
        
        
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1
# streamlit run Streamlit_RAG.py
