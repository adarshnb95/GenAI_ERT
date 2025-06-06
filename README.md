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
2. [Folder Structure](#folder-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Environment Variables](#environment-variables)
5. [Usage](#usage)
6. [5-Day Development Plan](#5-day-development-plan)
7. [Adding New Question Types](#adding-new-question-types)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)

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

  * `run_pipeline.py`: Automates EDGAR ingestion & FAISS index builds.
  * `start_app.py`: Launches both FastAPI and Streamlit concurrently.

---

## Folder Structure

```
GenAI_ERT/
├── api/
│   ├── ask_handlers.py       # Handler classes for question routing
│   └── main.py               # FastAPI app (factory mode)
├── ingestion/
│   ├── edgar_fetch.py        # Fetch filings per ticker
│   ├── news_fetch.py         # (Optional) Fetch news via NewsAPI
│   ├── data/                 # Raw filings stored per ticker
│   └── news_data/            # Raw news JSON per ticker
├── summarization/
│   ├── summarize.py          # RAG logic, build/retrieve FAISS index
│   ├── extract_metrics.py     # XBRL parsing helpers
│   ├── news_index.py         # Build/retrieve news FAISS index
│   └── faiss_index/          # Per-ticker FAISS indexes & metadata
├── tests/                    # Unit & end-to-end tests
├── dashboard_app.py          # Streamlit frontend
├── run_pipeline.py           # Ingest → Index automation
├── start_app.py              # Launch backend + frontend
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

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
  ```

from ingestion.edgar\_fetch import fetch\_for\_ticker
from summarization.summarize import build\_faiss\_index\_for\_ticker
fetch\_for\_ticker("AAPL")
build\_faiss\_index\_for\_ticker("AAPL", reset=True)
EOF

````
- **Build news index**:
```bash
python - <<'EOF'
from ingestion.news_fetch import fetch_and_store_news
from summarization.news_index import build_news_index
fetch_and_store_news("AAPL")
build_news_index(reset=True)
EOF
````

* **Run backend only**:

  ```bash
  uvicorn "api.main:create_app" --reload --reload-dir api --reload-dir summarization
  ```
* **Run frontend only**:

  ```bash
  streamlit run dashboard_app.py
  ```

---

## 5-Day Development Plan

**Day 1: Automate Ingestion & Indexing**

* Finalize `run_pipeline.py`, test end-to-end
* Commit automation script and verify folder outputs

**Day 2: Parameterize by Ticker**

* Refactor `ingestion/edgar_fetch.py`, `summarization/summarize.py` to accept ticker
* Verify `fetch_for_ticker("AAPL")` writes to `ingestion/data/AAPL/`
* Verify `build_faiss_index_for_ticker("AAPL", reset=True)` creates FAISS index

**Day 3: Unified Startup & UI Ticker Input**

* Create `start_app.py` to launch both services
* Update Streamlit (`dashboard_app.py`) to include ticker input
* Adapt FastAPI `/ask` to consume `ticker` from request

**Day 4: Handler-Based Routing & Tests**

* Implement `api/ask_handlers.py` with modular handlers
* Refactor `/ask` in `api/main.py` to loop through `ASK_HANDLERS`
* Write unit tests for each handler with TestClient

**Day 5: Performance & Polish**

* Lazy-load heavy models and FAISS imports
* Switch to Uvicorn factory mode and limit reload dirs
* Add end-to-end smoke tests (`tests/test_e2e.py`)
* Update README, finalize documentation

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

## Contributing

1. Fork and create a feature branch.
2. Install dependencies and run tests.
3. Add handlers or features following existing patterns.
4. Submit a pull request with descriptions.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
