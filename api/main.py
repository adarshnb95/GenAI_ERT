from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import re
from summarization.summarize import build_faiss_index_for_ticker, summarize_text, retrieve_context_for_ticker
# Ingestion functions
from ingestion.edgar_fetch import get_latest_filings, download_filing_index, choose_and_download, fetch_for_ticker
# Classification utility
from classifier.predict import classify_text
from api.ask_handlers import ASK_HANDLERS
from api.utils import extract_tickers_via_gpt

# Summarization placeholder (to be implemented)
# from summarization.summarize import summarize_text

app = FastAPI(
    title="Generative AI Equity Research Tool API",
    description="Endpoints for ingestion, classification, and summarization",
    version="0.1.0"
)

class IngestRequest(BaseModel):
    ticker: str
    count: int = 2  # number of recent filings to ingest

class ClassifyRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    text: str

TICKER_YEAR_PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?(20\d{2})\b")

@app.post("/ingest")
async def ingest(request: IngestRequest):
    cik = None
    try:
        cik = get_latest_filings.__globals__['TICKER_CIK'][request.ticker.upper()]
    except KeyError:
        raise HTTPException(status_code=404, detail="Ticker not found")

    results = []
    filings = get_latest_filings(cik, count=request.count)
    for f in filings:
        basename = f['filename']
        idx_path = download_filing_index(cik, f['accession'], basename)
        component_name = choose_and_download(cik, f['accession'], idx_path)
        results.append({
            "form": f['form'],
            "date": f['date'],
            "component": component_name
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
    tickers = extract_tickers_via_gpt(question)
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