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

# Summarization placeholder (to be implemented)
# from summarization.summarize import summarize_text

app = FastAPI(
    title="Generative AI Equity Research Tool API",
    description="Endpoints for ingestion, classification, and summarization",
    version="0.1.0"
)

class IngestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 'AAPL'")
    count: int = Field(20, description="Max number of filings to fetch")
    form_types: Optional[List[str]] = Field(
        None,
        description="List of SEC form types, e.g. ['10-K','N-2']; defaults to ['10-K','10-Q']"
    )

class ClassifyRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    text: str

TICKER_YEAR_PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?(20\d{2})\b")

@app.post("/ingest")
async def ingest(req: IngestRequest):
    ticker = req.ticker.upper()
    count  = req.count
    forms  = tuple(req.form_types) if req.form_types else ("10-K","10-Q")

    # 1) Fetch filings (does dynamic CIK lookup & download)
    try:
        index_paths = fetch_for_ticker(ticker, count=count, form_types=forms)
    except ValueError as e:
        # e.g. unknown ticker
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

    # 2) Build/update FAISS index (non-blocking)
    try:
        build_faiss_index_for_ticker(ticker, reset=False)
    except Exception:
        # log but don’t fail the ingest call
        print(f"[ingest] could not build FAISS index for {ticker}")

    # 3) Summarize what got saved
    results = []
    for idx in index_paths:
        stem   = Path(idx).stem  # e.g. "PTY-000012345678-index"
        folder = Path(idx).parent
        comps  = [p.name for p in folder.iterdir()
                  if p.name.startswith(stem) and p.name != Path(idx).name]
        results.append({
            "index_file": Path(idx).name,
            "components": comps
        })

    return {"ingested": results}

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
    question = req.text.strip()

    # 1) Use GPT to pull company names → tickers
    tickers = extract_tickers_from_text(question)
    if not tickers:
        raise HTTPException(400, "Couldn't identify any known companies in your question.")

    # 2) Ensure data/index exist for each ticker
    for ticker in tickers:
        fetch_for_ticker(ticker)
        build_faiss_index_for_ticker(ticker, reset=False)

    # 3) Dispatch to handlers (each now accepts List[str], text)
    for handler in ASK_HANDLERS:
        if handler.can_handle(question):
            return handler.handle(tickers, question)

    raise HTTPException(500, "No handler could process the question.")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)