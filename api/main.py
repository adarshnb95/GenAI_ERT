from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import uvicorn
import re
from summarization.summarize import build_faiss_index_for_ticker, summarize_text, retrieve_context_for_ticker
# Ingestion functions
from ingestion.edgar_fetch import get_latest_filings, choose_and_download, fetch_for_ticker
# Classification utility
from classifier.predict import classify_text
from api.ask_handlers import ASK_HANDLERS
from api.utils import extract_tickers_from_text
from starlette.concurrency import run_in_threadpool
from functools import partial
from pydantic import BaseModel

# Summarization placeholder (to be implemented)
# from summarization.summarize import summarize_text

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


@app.post("/ask")
async def ask(req: AskRequest):
    # 1) Grab the question text
    question = req.text.strip()

    # 2) Find tickers (your existing logic)
    tickers = extract_tickers_from_text(question)
    if not tickers:
        raise HTTPException(400, "No tickers found")

    # 3) Kick off ingestion and indexing in threads
    for ticker in tickers:
        # fetch_for_ticker and build_faiss_index_for_ticker are blocking I/O
        await run_in_threadpool(
            partial(fetch_for_ticker, ticker,
                    count=req.count,
                    form_types=tuple(req.form_types))
        )
        await run_in_threadpool(
            partial(build_faiss_index_for_ticker, ticker, reset=False)
        )

    # 4) Now dispatch to your RAG/metrics handlers
    for handler in ASK_HANDLERS:
        if handler.can_handle(question):
            return handler.handle(tickers, question)

    raise HTTPException(500, "No handler could process the question.")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)