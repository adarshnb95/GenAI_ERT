import requests
from transformers import pipeline

# initialize once at import
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_headlines(ticker, api_key="<YOUR_NEWSAPI_KEY>"):
    url = ("https://newsapi.org/v2/everything"
           f"?q={ticker}&sortBy=publishedAt&apiKey={api_key}")
    resp = requests.get(url)
    resp.raise_for_status()
    return [art["title"] for art in resp.json().get("articles", [])]

def analyze_sentiment(texts):
    return sentiment_pipe(texts)