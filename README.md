# Generative AI Equity Research Tool (GenAI\_ERT)

A proof-of-concept pipeline that:

* Ingests the latest SEC filings (10-K, 10-Q) for public companies via EDGAR
* Classifies documents using a fine-tuned DistilBERT model
* (Future) Summarizes filings and answers free-text questions with a retrieval-augmented generation (RAG) approach
* Exposes functionality through a FastAPI service

---

## 🚀 Features

* **Ingestion**: Download and parse EDGAR filings (XBRL or HTML).
* **Classification**: Auto-label filings (`10-K`, `10-Q`, `Transcript`, etc.) with a transformer classifier.
* **API**: REST endpoints for ingestion and classification (Swagger UI at `/docs`).
* **Modular**: Easily extendable for summarization, embedding/RAG, sentiment analysis, and front-end integration.

---

## 📂 Repository Structure

```
GenAI_ERT/
├── api/
│   └── main.py               # FastAPI application
├── ingestion/
│   └── edgar_fetch.py        # EDGAR ingestion and download logic
├── classifier/
│   ├── checkpoint/           # Transformer model & tokenizer artifacts (ignored)
│   ├── data/
│   │   └── labels.csv        # Labeled files for fine-tuning
│   ├── predict.py            # Classification helper
│   └── train_classifier.py   # Fine-tune DistilBERT on local labels
├── summarization/            # (Planned) RAG & summarization modules
├── tests/                    # Unit & integration tests
├── .gitignore
├── generate_labels.py        # Auto-generate labels template
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

* Python 3.9+ (virtual environment recommended)
* Git
* (Optional) Docker, if containerizing services

---

## 🛠️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/GenAI_ERT.git
   cd GenAI_ERT
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare ingestion data** (optional)

   ```bash
   python generate_labels.py
   ```

5. **Fine-tune classifier** (after populating `classifier/data/labels.csv`)

   ```bash
   python classifier/train_classifier.py
   ```

6. **Test the trained model**

   ```bash
   python test_model.py
   ```

7. **Run the API**

   ```bash
   pip install fastapi uvicorn
   uvicorn api.main:app --reload
   ```

Visit `http://127.0.0.1:8000/docs` for interactive API docs. (Currently unavailable)

---

## 📈 Usage Examples

* **Ingest filings for AAPL**

  ```bash
  curl -X POST "http://127.0.0.1:8000/ingest" \
    -H "Content-Type: application/json" \
    -d '{"ticker": "AAPL", "count": 2}'
  ```

* **Classify a text snippet**

  ```bash
  curl -X POST "http://127.0.0.1:8000/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": "In Q3, revenue grew 10% YoY."}'
  ```

---

## Tips

1. I noticed that the SEC files had already classified the files as 10-K and 10-Q, so I created a generate_labels.py that helped me generate test data using the official classification.

---

## 🤝 Contributing

Feel free to open issues or submit PRs. Please ensure new features have corresponding tests under `tests/`.

---

## 📜 License

[MIT License](LICENSE)
