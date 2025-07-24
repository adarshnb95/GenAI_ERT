import yfinance as yf

def get_price_series(ticker, period="1mo", interval="1d"):
    return yf.Ticker(ticker).history(period=period, interval=interval)

def compute_sma(series, window):
    return series["Close"].rolling(window).mean().iloc[-1]