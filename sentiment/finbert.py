from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT (huggingface “yiyanghkust/finbert-tone”)
MODEL_NAME = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone",
    top_k=1
)

def sentiment_score(text: str) -> dict:
    """
    Returns a dict with 'label' (Positive/Negative/Neutral)
    and 'score' (confidence) for the given text.
    """
    result = sentiment(text[:512])[0]   # limit to 512 tokens
    return {"label": result["label"], "score": result["score"]}
