import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from datasets import Dataset
import evaluate

# Paths
LABELS_CSV = os.path.join(os.path.dirname(__file__), 'data', 'labels.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')

# Load CSV and filter out unlabeled rows
df = pd.read_csv(LABELS_CSV)
df = df[df['label'].notna() & (df['label'] != '')]

# Read file contents
texts = []
labels = []
for _, row in df.iterrows():
    path = os.path.join(os.path.dirname(__file__), '..', row['filepath'])
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read(2000))  # read first 2000 chars
        labels.append(row['label'])
    except FileNotFoundError:
        continue

# Encode labels
label_list = sorted(list(set(labels)))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
encoded_labels = [label2id[l] for l in labels]

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

# Tokenizer & Dataset
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
dataset = dataset.map(tokenize_fn, batched=True)
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Model
def model_init():
    return DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

# Training arguments (minimal compatibility)
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Metrics
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

# Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")
