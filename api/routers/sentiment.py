from fastapi import APIRouter
import yfinance as yf

router = APIRouter(prefix="/technicals")

def get_price_series(ticker, period="1mo", interval="1d"):
    return yf.Ticker(ticker).history(period=period, interval=interval)

@router.get("/{ticker}/sma/{window}")
def get_sma(ticker: str, window: int):
    df = get_price_series(ticker)
    sma = df["Close"].rolling(window).mean().iloc[-1]
    return {"ticker": ticker, "window": window, "sma": sma}