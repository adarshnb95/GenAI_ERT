import sys, os

# Add the parent of the scripts folder (your project root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarization.extract_metrics import get_revenue_by_year, get_net_income_by_year

if __name__ == "__main__":
    ticker = "AAPL"
    for year in [2018, 2019, 2020, 2021]:
        rev = get_revenue_by_year(ticker, year)
        ni  = get_net_income_by_year(ticker, year)
        print(f"{ticker} {year:>4} → Revenue: {rev or 'n/a'}, Net Income: {ni or 'n/a'}")