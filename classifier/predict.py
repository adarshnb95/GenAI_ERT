from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load once at import time
MODEL_PATH = 'classifier/checkpoint'
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def classify_text(text: str) -> str:
    """Return the classifier’s predicted label for the given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]
