from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT (huggingface “yiyanghkust/finbert-tone”)
MODEL_NAME = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False    # returns the single best label
)

def sentiment_score(text: str) -> dict:
    """
    Returns a dict with 'label' (Positive/Negative/Neutral)
    and 'score' (confidence) for the given text.
    """
    result = sentiment_pipe(text[:512])[0]   # limit to 512 tokens
    return {"label": result["label"], "score": result["score"]}
