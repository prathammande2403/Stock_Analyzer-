import os
import json
import httpx
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Base URLs
BASE_URL_ALPHA = "https://www.alphavantage.co/query"
BASE_URL_NEWS = "https://newsapi.org/v2/everything"


# ------------------ Fetch Functions ------------------ #

def fetch_alpha(symbol: str):
    """Fetch intraday stock data from Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
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


# ------------------ Data Collection ------------------ #

def collect_stock_data(symbol: str):
    """Collect Alpha Vantage + NewsAPI data and save to JSON"""
    data = {
        "alpha_vantage": fetch_alpha(symbol),
        "news_api": fetch_news(symbol)
    }

    # Save combined JSON
    os.makedirs("data", exist_ok=True)
    filename = os.path.join("data", f"{symbol}_combined.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Combined data for {symbol} saved to {filename}")
    return data


# ------------------ Preprocessing ------------------ #

def prepare_llm_input(data: dict, symbol: str):
    """Extract recent candles and headlines to reduce token size"""
    # Get latest 5 candles
    time_series = data["alpha_vantage"].get("Time Series (5min)", {})
    latest_candles = list(time_series.items())[:5]

    # Get top 5 news headlines
    news_articles = data["news_api"].get("articles", [])[:5]
    headlines = [a.get("title", "") for a in news_articles]

    return {
        "symbol": symbol,
        "latest_candles": latest_candles,
        "headlines": headlines
    }


# ------------------ LLM Setup ------------------ #

def analyze_stock(symbol: str):
    """Run full pipeline and get BUY/SELL/HOLD suggestion from LLM"""
    # Step 1: Collect raw data
    raw_data = collect_stock_data(symbol)

    # Step 2: Prepare trimmed input for LLM
    llm_input = prepare_llm_input(raw_data, symbol)

    # Step 3: Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0.3
    )

    # Step 4: Define Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant that gives BUY, SELL, or HOLD recommendations."),
        ("human", """
        Stock Symbol: {symbol}
        
        Latest Candles (Time, OHLCV): {latest_candles}
        
        Recent News Headlines: {headlines}
        
        Based on stock data and news sentiment, suggest one clear action:
        BUY, SELL, or HOLD. 
        Also provide a brief reasoning (2-3 sentences).
        """)
    ])

    parser = StrOutputParser()

    # Step 5: Format prompt
    prompt = prompt_template.format(**llm_input)

    # Step 6: Get LLM Response
    response = llm.invoke(prompt)
    result = parser.parse(response.content)

    print("📊 LLM Decision:\n", result)
    return result


# ------------------ Run Example ------------------ #

if __name__ == "__main__":
    analyze_stock("AAPL")  # Example: Apple stock

