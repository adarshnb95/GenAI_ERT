from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import uvicorn
import re
from summarization.summarize import build_faiss_index_for_ticker, summarize_text, retrieve_context_for_ticker
# Ingestion functions
from ingestion.edgar_fetch import (
    get_latest_filings, 
    choose_and_download, 
    fetch_for_ticker, 
    fetch_financial_fact,
    get_cik_for_ticker,
    fetch_xbrl_for_year,
    extract_fact_request
)
# Classification utility
from classifier.predict import classify_text
from api.ask_handlers import ASK_HANDLERS
from api.utils import extract_tickers_from_text
from starlette.concurrency import run_in_threadpool
from functools import partial
from pydantic import BaseModel
import logging

from api.routers import fundamentals, technicals, sentiment

app = FastAPI()

app.include_router(fundamentals.router)
app.include_router(technicals.router)
app.include_router(sentiment.router)

# Summarization placeholder (to be implemented)
# from summarization.summarize import summarize_text

METRIC_LABELS = {
    "NetIncomeLoss": "net income",
    "Revenues": "revenue",
    "OperatingIncomeLoss": "operating income",
    "NetCashProvidedByUsedInOperatingActivities": "operating cash flow",
    "EarningsPerShareBasic": "basic EPS",
    "Assets": "total assets",
    "Liabilities": "total liabilities",
    "StockholdersEquity": "shareholders’ equity",
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Generative AI Equity Research Tool API",
    description="Endpoints for ingestion, classification, and summarization",
    version="0.1.0"
)

class IngestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 'AAPL'")
    count: int = Field(50, description="Max number of filings to fetch")
    form_types: Optional[List[str]] = Field(
        None,
        description="List of SEC form types, e.g. ['10-K','N-2']; defaults to ['10-K','10-Q']"
    )

class ClassifyRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    text: str
    count: int = 20
    form_types: List[str] = ["10-K", "10-Q"]

TICKER_YEAR_PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?(20\d{2})\b")

from api.ask_handlers import SimpleMetricHandler
SIMPLE_HANDLERS = [
    # SimpleRevenueHandler(),
    SimpleMetricHandler(),
    # …any other lightweight handlers…
]

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Only do the EDGAR download part. Don’t touch FAISS or embeddings here.
    """
    try:
        paths = await run_in_threadpool(
            fetch_for_ticker,
            request.ticker,
            request.count,
            tuple(request.form_types) if request.form_types else ("10-K", "10-Q")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    s3_keys = []
    for path in paths:
        name = path.name                     # e.g. "AAPL-0001-index.json" or "AAPL-0001.xml"
        ticker, rest = name.split("-", 1)    # ticker="AAPL", rest="0001-index.json" or "0001.xml"

        if rest.startswith("0000") or "-" in rest:
            # This covers the index.json case ("0001-index.json")
            accession, fname = rest.split("-", 1)
        else:
            # This covers the simple XML case ("0001.xml")
            accession = rest.split(".", 1)[0]  # "0001"
            fname = name                       # full filename

        s3_keys.append(f"edgar/{ticker}/{accession}/{fname}")
    
    return {"ingested_s3_keys": s3_keys}

class BuildIndexRequest(BaseModel):
    ticker: str
    reset: bool = False

@app.post("/build_index")
async def build_index(req: BuildIndexRequest):
    ticker = req.ticker.upper()
    await run_in_threadpool(
        partial(build_faiss_index_for_ticker, ticker, reset=req.reset)
    )
    return {"status": "index built", "ticker": ticker}

@app.post("/classify")
async def classify(request: ClassifyRequest):
    try:
        label = classify_text(request.text)
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/summarize")
async def summarize(request: ClassifyRequest):
    # Re-build index if needed (or schedule separately)
    # build_faiss_index(reset=False)
    summary = summarize_text(request.text)
    return {"summary": summary}

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

HELP_PATTERNS = [
    r"\bhelp\b",
    r"\bwhat can you do\b",
    r"\bwhat information can you give\b",
    r"\bhow (do|can) i (use|ask)\b",
    r"\bwhat are your capabilities\b",
    r"\btell me about( yourself)?\b",
    r"\bhow (does|do) this work\b",
    r"\busage\b",
    r"\bguide\b",
]

HELP_ANSWER = (
    "I can:\n"
    "- Instantly fetch single-value facts (revenue, net income, EPS, assets, etc.) via the SEC’s XBRL API\n"
    "- Ingest and index SEC filings (10-K, 10-Q, N-2) into a FAISS search engine\n"
    "- Answer deeper “why” or “explain” questions with classification, embedding, and RAG-driven summarization\n"
    "- Streamline simple metric queries for speed, and fall back to full NLP only when needed\n\n"
    "Try asking things like:\n"
    "• “What did Apple earn in 2020?”\n"
    "• “Explain why revenue grew in Q2.”\n"
    "• “Show me Apple’s asset breakdown.”"
)

@app.post("/ask")
async def ask(req: AskRequest):
    question = req.text.strip()
    logger.info(f"[ask] incoming question: {question!r}")

    # 1) detect tickers
    tickers = extract_tickers_from_text(question)
    if not tickers:
        raise HTTPException(400, "No tickers found in your question.")
    logger.info(f"[ask] detected tickers: {tickers}")

    # 1b) generic “help” / meta questions
    for pat in HELP_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            return {"answer": HELP_ANSWER}

    # 2) fact-check branch (any single-value XBRL metric + year)
    metric_key, year, period = extract_fact_request(question)
    if tickers and metric_key and year:
        label = METRIC_LABELS.get(metric_key, metric_key)
        logger.info(f"[ask] fact-check lookup: {metric_key} for {tickers[0]} in {year}")
        cik = get_cik_for_ticker(tickers[0])
        if not cik:
            raise HTTPException(404, f"CIK not found for {tickers[0]}")
        try:
            value = fetch_financial_fact(cik, metric_key, year, period)
            return {"answer": f"{tickers[0]}'s {label} in {period} {year} was ${value:,}"}
        except KeyError as e:
            raise HTTPException(404, str(e))

    # 3) fallback to any other simple-metric handler you might still have
    if SimpleMetricHandler.can_handle(question):
        logger.info("[ask] matched SimpleMetricHandler — skipping FAISS/classifier.")
        return SimpleMetricHandler.handle(tickers, question)

    # 4) ingestion + indexing
    for ticker in tickers:
        await run_in_threadpool(
            partial(fetch_for_ticker, ticker,
                    count=req.count,
                    form_types=tuple(req.form_types))
        )
        await run_in_threadpool(
            partial(build_faiss_index_for_ticker, ticker, reset=False)
        )

    # 5) remaining handlers (classifier/RAG)
    for handler in ASK_HANDLERS:
        if handler.can_handle(question):
            logger.info(f"[ask] dispatching to {handler.__class__.__name__}")
            return handler.handle(tickers, question)

    raise HTTPException(500, "Unable to handle the question.")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)