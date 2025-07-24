from fastapi import APIRouter, HTTPException
from ingestion.edgar_fetch import get_cik_for_ticker, fetch_financial_fact

router = APIRouter(prefix="/fundamentals")

@router.get("/{ticker}/{metric}/{year}")
def get_fact(ticker: str, metric: str, year: int):
    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise HTTPException(404, f"CIK not found for {ticker}")
    val = fetch_financial_fact(cik, metric, year)
    return {"ticker": ticker, "metric": metric, "year": year, "value": val}