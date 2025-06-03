from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import re
from summarization.summarize import build_faiss_index, summarize_text, answer_question
from summarization.extract_metrics import (
    get_latest_net_income,
    get_latest_revenue,
    get_revenue_by_year
)
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
    q_raw = request.text.strip()
    q = q_raw.lower()
    print(f"[DEBUG] Received /ask: '{q_raw}'")

    # 1) Net Income logic (unchanged)
    if "net income" in q:
        net_val = get_latest_net_income()
        print(f"[DEBUG] get_latest_net_income() returned: '{net_val}'")
        if net_val:
            return {"answer": f"${net_val}"}

    # 2) Revenue comparisons by year
    #    If the question mentions two or more years, extract them and compare
    years = YEAR_PATTERN.findall(q)  # e.g. ["2022", "2023"]
    if "revenue" in q or "sales" in q:
        if years:
            # If user asked “What was revenue in 2022? Compared with 2023?”
            # We handle up to two years here
            responses = []
            for yr in years[:2]:  # take the first two years mentioned
                rev = get_revenue_by_year(yr)
                if rev:
                    responses.append(f"{yr}: ${rev}")
                else:
                    responses.append(f"{yr}: Not found")
            # If exactly two years, compare
            if len(responses) == 2:
                return {"answer": f"{responses[0]} vs {responses[1]}"}
            # Otherwise return whatever we found
            return {"answer": "; ".join(responses)}

        # If no specific year was mentioned, fall back to “latest only”
        rev_val = get_latest_revenue()
        print(f"[DEBUG] get_latest_revenue() returned: '{rev_val}'")
        if rev_val:
            return {"answer": f"${rev_val}"}

    # 3) Fallback to RAG
    try:
        answer = answer_question(q_raw)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
