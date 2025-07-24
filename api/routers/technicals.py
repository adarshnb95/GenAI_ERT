from fastapi import APIRouter
from analysis.news_sentiment import get_headlines, analyze_sentiment

router = APIRouter(prefix="/sentiment")

@router.get("/{ticker}")
def news_sentiment(ticker: str):
    headlines = get_headlines(ticker)
    scores = analyze_sentiment(headlines[:5])
    return {"ticker": ticker, "headlines": headlines[:5], "sentiment": scores}