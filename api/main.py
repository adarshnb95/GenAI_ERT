from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from summarization.summarize import build_faiss_index, summarize_text, answer_question

# Ingestion functions
from ingestion.edgar_fetch import get_latest_filings, download_filing_index, choose_and_download
# Classification utility
from classifier.predict import classify_text

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

@app.post("/ask")
async def ask(request: ClassifyRequest):
    answer = answer_question(request.text)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
