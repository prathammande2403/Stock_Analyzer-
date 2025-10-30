import os
import json
import httpx
import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ------------------ Setup ------------------ #
load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_URL_ALPHA = "https://www.alphavantage.co/query"
BASE_URL_NEWS = "https://newsapi.org/v2/everything"


# ------------------ Fetch Functions ------------------ #

def fetch_alpha_daily(symbol: str):
    """Fetch daily stock data from Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_API_KEY
    }
    response = httpx.get(BASE_URL_ALPHA, params=params, timeout=20.0)
    return response.json()


def fetch_news(symbol: str):
    """Fetch recent news for the stock/company"""
    params = {
        "q": symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY
    }
    response = httpx.get(BASE_URL_NEWS, params=params, timeout=20.0)
    return response.json()


# ------------------ Data Processing ------------------ #

def compute_technical_indicators(time_series: dict):
    """Compute basic trend + SMA + RSI from Alpha Vantage data"""
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    df['SMA_5'] = df['4. close'].rolling(window=5).mean()
    df['SMA_20'] = df['4. close'].rolling(window=20).mean()

    # RSI calculation
    delta = df['4. close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna()
    recent = df.iloc[-1]
# Use last 7 days to calculate trend
    last_days = df.tail(7)
    start, end = last_days['4. close'].iloc[0], last_days['4. close'].iloc[-1]
    pct_change = ((end - start) / start) * 100


    return {
        "trend": f"{pct_change:.2f}%",
        "sma5": recent['SMA_5'],
        "sma20": recent['SMA_20'],
        "rsi": recent['RSI']
    }


def analyze_news_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for a in articles:
        title = a.get("title","")
        if title:
            scores.append(analyzer.polarity_scores(title)["compound"])
    if not scores:
        return {"avg_sentiment": 0.0, "label": "neutral"}
    avg_sentiment = sum(scores)/len(scores)
    label = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
    return {"avg_sentiment": avg_sentiment, "label": label}


# ------------------ LLM Analysis ------------------ #

def analyze_stock(symbol: str):
    """Full pipeline with technical + sentiment + LLM reasoning"""
    st.write(f"\n📈 Analyzing {symbol}...")

    # Step 1: Fetch data
    alpha_data = fetch_alpha_daily(symbol)
    news_data = fetch_news(symbol)

    time_series = alpha_data.get("Time Series (Daily)", {})
    articles = news_data.get("articles", [])[:5]

    if not time_series:
        st.write("⚠️ No stock data found!")
        return

    # Step 2: Preprocess
    indicators = compute_technical_indicators(time_series)
    sentiment = analyze_news_sentiment(articles)
    headlines = [a.get("title", "") for a in articles]

    # Step 3: Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0.3
    )

    # Step 4: Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a financial advisor that decides between BUY, SELL, or HOLD.
Use these rules:
- If trend is positive (> +2%) and sentiment is positive → BUY.
- If trend is negative (< -2%) and sentiment is negative → SELL.
- Otherwise → HOLD.
Use RSI to check momentum (RSI > 70 = overbought, RSI < 30 = oversold).
Compare SMA_5 vs SMA_20 to see trend direction.
Give one clear decision (BUY, SELL, or HOLD) and explain briefly.
"""),
        ("human", """
Stock Symbol: {symbol}
Trend over period: {trend}
RSI: {rsi:.2f}
SMA(5): {sma5:.2f}, SMA(20): {sma20:.2f}
Average News Sentiment: {sentiment_label} ({avg_sentiment:.2f})
Recent Headlines: {headlines}
""")
    ])

    parser = StrOutputParser()

    # Step 5: Prepare inputs
    prompt = prompt_template.format(
        symbol=symbol,
        trend=indicators["trend"],
        rsi=indicators["rsi"],
        sma5=indicators["sma5"],
        sma20=indicators["sma20"],
        sentiment_label=sentiment["label"],
        avg_sentiment=sentiment["avg_sentiment"],
        headlines=headlines
    )

    # Step 6: Get Response
    response = llm.invoke(prompt)
    result = parser.parse(response.content)

    st.write("\n📊\n", result)
    return result


# ------------------ Run Example ------------------ #

if __name__ == "__main__":
    analyze_stock("TSLA")   # You can replace with any symbol (e.g., "TSLA", "GOOG")
