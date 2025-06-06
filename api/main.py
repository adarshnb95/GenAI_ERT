from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import re
from summarization.summarize import build_faiss_index, summarize_text
# Ingestion functions
from ingestion.edgar_fetch import get_latest_filings, download_filing_index, choose_and_download
# Classification utility
from classifier.predict import classify_text
from api.ask_handlers import ASK_HANDLERS

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

TICKER_YEAR_PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?(20\d{2})\b")

@app.on_event("startup")
async def startup_event():
    # Build (or reload) the FAISS index before serving any requests
    build_faiss_index(reset=False)

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
async def ask(request: ClassifyRequest):
    """
    Route the incoming question to the first handler whose `can_handle`
    returns True. If it throws an exception, propagate it as HTTP 500.
    """
    q = request.text.strip()
    print(f"[DEBUG] /ask received: '{q}'")

    for handler in ASK_HANDLERS:
        try:
            if handler.can_handle(q):
                return handler.handle(q)
        except HTTPException:
            # If a handler explicitly raises HTTPException, bubble it up
            raise
        except Exception as e:
            # Any other exception—wrap in HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    # In theory, RAGFallbackHandler always handles last,
    # so we should never reach here.
    raise HTTPException(status_code=500, detail="Unable to handle the question.")



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
