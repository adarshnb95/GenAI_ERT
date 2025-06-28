from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load your trained model & tokenizer
model = DistilBertForSequenceClassification.from_pretrained(
    "./models/classifier-checkpoint"
)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Prepare a sample input
text = "In Q2, revenue rose by 15% year-over-year thanks to strong product sales."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

# Run inference
with torch.no_grad():
    logits = model(**inputs).logits
pred_id = logits.argmax(dim=-1).item()

# Print out the predicted label
print("Predicted label:", model.config.id2label[pred_id])
