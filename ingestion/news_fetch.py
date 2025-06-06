import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from newsapi import NewsApiClient

# 1) Configure your NewsAPI.org key as an environment variable
load_dotenv()  # Load environment variables from .env file
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise RuntimeError("Please set the NEWSAPI_KEY environment variable with your NewsAPI.org key.")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Where we’ll store raw news articles
NEWS_DATA_DIR = Path(__file__).resolve().parent / "news_data"
NEWS_DATA_DIR.mkdir(exist_ok=True)

def fetch_and_store_news(ticker: str, page_size: int = 100) -> List[Path]:
    """
    Fetches the latest news articles mentioning 'ticker' (e.g. "Apple Inc"), stores each
    as a JSON file under ingestion/news_data/, and returns the list of file paths.
    """
    # Example: query for "{ticker} stock OR {ticker} revenue" to bias financial news
    query = f"{ticker} stock OR {ticker} revenue OR {ticker} earnings"
    all_articles = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=page_size
    )

    saved_paths = []
    for art in all_articles.get("articles", []):
        # Build a minimal record: headline, description, url, publishedAt
        rec = {
            "source": art["source"]["name"],
            "title": art["title"] or "",
            "description": art["description"] or "",
            "url": art["url"],
            "publishedAt": art["publishedAt"]
        }
        # Filename by published date + ticker, e.g. "AAPL-20240510T143212.json"
        dt = rec["publishedAt"].replace(":", "").replace("-", "").replace("Z", "")
        fn = f"{ticker.upper()}-{dt}.json"
        out_path = NEWS_DATA_DIR / fn
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        saved_paths.append(out_path)

    print(f"Fetched {len(saved_paths)} articles for {ticker}.")
    return saved_paths

if __name__ == "__main__":
    # Example: fetch for Apple, Microsoft, Salesforce
    for tk in ("AAPL", "MSFT", "CRM"):
        fetch_and_store_news(tk)
