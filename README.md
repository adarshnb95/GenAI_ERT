# Generative AI Equity Research Tool

A full-stack Python application for automated EDGAR filings ingestion, FAISS-based retrieval, and LLM-enhanced equity research. This project lets you:

* Fetch and store 10-K/10-Q filings (and optional news) for any public ticker
* Build per-ticker FAISS indexes over document chunks (with sentiment analysis)
* Extract and compare financial metrics (net income, revenue, profit percentage) across tickers and years
* Provide forward-looking “stock outlook” answers by combining latest metrics with recent news
* Serve a Streamlit dashboard (port 8501) and FastAPI backend (port 8000) in one click

---

## Table of Contents

1. [Features](#features)
2. [Folder and Design Structure](#folder-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Environment Variables](#environment-variables)
5. [Usage](#usage)
6. [5-Day Development Plan](#5-day-development-plan)
7. [Adding New Question Types](#adding-new-question-types)
8. [Testing](#testing)
9. [Notes](#notes)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features

* **EDGAR Ingestion**: Download and store the latest 10-K and 10-Q filings (JSON index, XBRL, HTML) under `ingestion/data/<TICKER>/`.
* **FAISS Indexing**: Split filings into fixed-size chunks, compute sentiment, embed with SentenceTransformer, and build per-ticker FAISS indexes.
* **Metric Extraction**:

  * `get_net_income_by_year(ticker, year)` & `get_revenue_by_year(ticker, year)`
  * `get_latest_net_income(ticker)` & `get_latest_revenue(ticker)`
  * Profit and profit-percentage comparisons via `get_net_income_by_years()` and `get_profit_percentage_by_years()`
* **News Integration (Optional)**: Fetch recent financial news via NewsAPI, embed title+description, and build a news FAISS index.
* **Ask–Handlers**: Modular handler classes for routing questions, including:

  * `NetIncomeYearHandler`, `LatestNetIncomeHandler`
  * `RevenueYearHandler`, `LatestRevenueHandler`
  * `ProfitCompareHandler`, `ProfitPctCompareHandler`
  * `StockNewsHandler` (combines metrics + news → GPT prompt)
  * `RAGFallbackHandler` (retrieval-augmented GPT over filings)
* **Streamlit Dashboard**: Interactive frontend for users to input ticker & question; displays answers.
* **FastAPI Backend**: REST API for ingestion, summarization, and QA endpoints via Uvicorn.
* **Automation Scripts**:

  * `start_app.py`: Launches both FastAPI and Streamlit concurrently.
  * `check_metrics.py`: Tests for Revenue and Income without using FAISS

---

## Folder Structure

```
GenAI_ERT/
├── api/
│   ├── main.py
│   ├── ask_handlers.py
│   └── utils.py
├── ingestion/
│   ├── edgar_fetch.py
│   ├── company_tickers.json  # fetched at runtime
│   └── data/
│       ├── AAPL/… 
│       ├── EOD/…
│       └── PTY/…
├── summarization/
│   ├── summarize.py
│   ├── extract_metrics.py
│   ├── news_index.py
│   └── faiss_index/
├── classifier/
│   ├── train_classifier.py
│   ├── predict.py
│   └── checkpoint/  # model.safetensors, tokenizer.json, …
├── dashboard/
│   └── dashboard_app.py
├── scripts/
│   └── inspect_index.py
├── start_app.py
├── README.md
├── requirements.txt
└── .gitignore
```
System design diagram:

![System_design_diagram](https://github.com/adarshnb95/GenAI_ERT/blob/main/design/design_diagram.png "System Design Diagram")


---

## Prerequisites & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/GenAI_ERT.git
   cd GenAI_ERT
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate     # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) System prerequisites**

   * Windows: install Visual C++ Build Tools for `faiss-cpu`.
   * macOS/Linux: FAISS CPU wheels install directly.

---

## Environment Variables

* **`OPENAI_API_KEY`**: Your OpenAI API key for GPT calls.

  ```bash
  export OPENAI_API_KEY="sk-…your_key…"   # macOS/Linux
  $Env:OPENAI_API_KEY="sk-…your_key…"     # PowerShell
  ```

* **`NEWSAPI_KEY`** *(optional)*: Your NewsAPI.org key for news ingestion.

  ```bash
  export NEWSAPI_KEY="your_news_key"       # macOS/Linux
  $Env:NEWSAPI_KEY="your_news_key"         # PowerShell
  ```

* **`.env` file**: Create `.env` at project root:

  ```ini
  OPENAI_API_KEY=sk-…your_key…
  NEWSAPI_KEY=your_news_key
  ```

  (Requires `python-dotenv`.)


Visit `http://127.0.0.1:8000/docs` for interactive API docs. 

---

## Usage

### 1. Run the Full Pipeline (Ingestion → Indexing)

Fetch filings & news (optional), then build FAISS indexes:

```bash
python run_pipeline.py
```

### 2. Start Backend & Frontend Together

```bash
python start_app.py
```

* **FastAPI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Streamlit**: [http://localhost:8501](http://localhost:8501)

### 3. Manual Steps

* **Ingest & index a single ticker**:

```bash
   python - <<'EOF'
   from ingestion.edgar\_fetch import fetch\_for\_ticker
   from summarization.summarize import build\_faiss\_index\_for\_ticker
   fetch\_for\_ticker("AAPL")
   build\_faiss\_index\_for\_ticker("AAPL", reset=True)
   EOF

```

* **Build news index**:

```bash
   python - <<'EOF'
   from ingestion.news_fetch import fetch_and_store_news
   from summarization.news_index import build_news_index
   fetch_and_store_news("AAPL")
   build_news_index(reset=True)
   EOF
```

* **Run backend only**:

  ```bash
   uvicorn "api.main:create_app" --reload --reload-dir api --reload-dir summarization
  ```

* **Run frontend only**:

  ```bash
   streamlit run dashboard_app.py
  ```

---

## Adding New Question Types

To add a new handler (e.g. EBITDA comparison):

1. In `api/ask_handlers.py`, subclass `AskHandler`, implement `can_handle()` & `handle()`.
2. Insert the new handler into `ASK_HANDLERS` before broader matches.
3. Implement corresponding helper in `summarization/extract_metrics.py`.

---

## Testing

Run unit tests and end-to-end smoke tests:

```bash
pytest -q tests/
```
---

## Notes

1. I noticed that the SEC files had already classified the files as 10-K and 10-Q, so I created a generate_labels.py that helped me generate test data using the official classification.
2. I created an extract_metrics.py file that helps in collecting the files required according to the question asked.
3. Some errors identified in the approach, so changing the structure of the layout and the way questions are answered. More logical answers will use the gpt enhanced FAISS/LLM whereas simple questions will just use XBRL parsing.

---

## Current status

Currently working on improving questioning and getting precise results.

---

## Contributing

1. Fork and create a feature branch.
2. Install dependencies and run tests.
3. Add handlers or features following existing patterns.
4. Submit a pull request with descriptions.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
