import re, datetime
import json
import openai
from fastapi import HTTPException
from typing import List
from pathlib import Path
from dotenv import load_dotenv


from ingestion.edgar_fetch import fetch_for_ticker
from summarization.summarize import (
    build_faiss_index_for_ticker,
    retrieve_context_for_ticker
)
from summarization.extract_metrics import (
    get_latest_revenue,
    get_revenue_by_year,
    get_latest_net_income,
    get_net_income_by_year,
    get_net_income_by_years,
    get_profit_percentage_by_years,
)
from summarization.news_index import retrieve_news_context
from summarization.summarize import answer_question_for_ticker, extract_n2_fields_from_text


class AskHandler:
    """
    Base class for `/ask` handlers. Subclasses implement:
      can_handle(text: str) -> bool
      handle(tickers: List[str], text: str) -> dict
    """
    def can_handle(self, text: str) -> bool:
        raise NotImplementedError

    def handle(self, tickers: List[str], text: str) -> dict:
        raise NotImplementedError


class NetIncomeYearHandler(AskHandler):
    PATTERN = re.compile(r"\bnet income\D+?(20\d{2})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, tickers: List[str], text: str) -> dict:
        year = self.PATTERN.search(text).group(1)
        results = {}
        for t in tickers:
            ni = get_net_income_by_year(t, year)
            if ni:
                results[t] = ni
        if not results:
            return {"answer": f"No net income data for {', '.join(tickers)} in {year}."}
        return {"answer": results}


class LatestNetIncomeHandler(AskHandler):
    KEYWORDS = ["latest net income", "net income"]

    def can_handle(self, text: str) -> bool:
        return (any(kw in text.lower() for kw in self.KEYWORDS)
                and not re.search(r"20\d{2}", text))

    def handle(self, tickers: List[str], text: str) -> dict:
        results = {}
        for t in tickers:
            ni = get_latest_net_income(t)
            if ni:
                results[t] = ni
        if not results:
            raise HTTPException(status_code=400, detail="No net income data available.")
        return {"answer": results}

class YearPerformanceHandler(AskHandler):
    # catch “revenue”, “perform(ed)”, “earnings”, “profit(s)”, etc. plus a year
    PATTERN = re.compile(
        r"\b(?:revenue|earnings|net income|perform(?:ed)?|how did)\b.*?\b(20\d{2})\b",
        re.IGNORECASE
    )

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, tickers: List[str], text: str) -> dict:
        import datetime
        from fastapi import HTTPException
        from ingestion.edgar_fetch import fetch_for_ticker
        from summarization.summarize import build_faiss_index_for_ticker
        from summarization.extract_metrics import get_revenue_by_year

        year   = int(self.PATTERN.search(text).group(1))
        ticker = tickers[0]

        # 1) compute how many filings to fetch
        current_year = datetime.date.today().year
        years_back   = current_year - year + 1
        count        = years_back * 5 + 3

        # 2) re-fetch & index
        fetched = fetch_for_ticker(ticker, count=count, form_types=("10-K","10-Q"))
        if not fetched:
            raise HTTPException(404, f"No filings for {ticker} back to {year}.")
        build_faiss_index_for_ticker(ticker, reset=True)

        # 3) extract the revenue
        rev = get_revenue_by_year(ticker, year)
        if not rev:
            return {"answer": f"No revenue data found for {ticker} in {year}."}
        return {"answer": {ticker: rev}}

class LatestRevenueHandler(AskHandler):
    KEYWORDS = ["latest revenue", "recent revenue", "revenue"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return (any(kw in txt for kw in self.KEYWORDS)
                and not re.search(r"20\d{2}", text))

    def handle(self, tickers: List[str], text: str) -> dict:
        results = {}
        for t in tickers:
            rev = get_latest_revenue(t)
            if rev:
                results[t] = rev
        if not results:
            raise HTTPException(status_code=400, detail="No revenue data available.")
        return {"answer": results}


class ProfitCompareHandler(AskHandler):
    def can_handle(self, text: str) -> bool:
        return "profit" in text.lower() and "than" in text.lower()

    def handle(self, tickers: List[str], text: str) -> dict:
        if len(tickers) < 2:
            raise HTTPException(status_code=400, detail="Please mention two companies to compare profits.")
        t1, t2 = tickers[:2]
        comp = get_net_income_by_years(t1, t2)
        for year, (ni1, ni2) in sorted(comp.items(), reverse=True):
            try:
                if int(ni1) > int(ni2):
                    return {"answer": {"year": year, t1: ni1, t2: ni2}}
            except:
                continue
        return {"answer": f"No year found where {t1} net income > {t2}."}


class ProfitPctCompareHandler(AskHandler):
    def can_handle(self, text: str) -> bool:
        return "profit percentage" in text.lower() and "than" in text.lower()

    def handle(self, tickers: List[str], text: str) -> dict:
        if len(tickers) < 2:
            raise HTTPException(status=400, detail="Mention two companies to compare profit percentages.")
        t1, t2 = tickers[:2]
        pct_map = get_profit_percentage_by_years(t1, t2)
        for year, (pct1, pct2) in sorted(pct_map.items(), reverse=True):
            if pct2 > pct1:
                return {"answer": {"year": year, t2: f"{pct2}%", t1: f"{pct1}%"}}
        return {"answer": f"No year found where {t2} profit percentage > {t1}."}


class StockNewsHandler(AskHandler):
    KEYWORDS = ["stock", "stocks", "news"]

    def can_handle(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.KEYWORDS)

    def handle(self, tickers: List[str], text: str) -> dict:
        results = {}
        for t in tickers:
            latest_rev = get_latest_revenue(t)
            try:
                news = retrieve_news_context(t, text, top_k=5)
            except:
                news = []
            joined = "\n\n---\n\n".join(news)
            prompt = (
                f"You are an equity research assistant. {t}’s most recent revenue was ${latest_rev}.\n\n"
                f"Recent news:\n{joined}\n\nQuestion: {text}"
            )
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You are a financial analyst."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            results[t] = resp.choices[0].message.content.strip()
        return {"answer": results}


class FormN2Handler(AskHandler):
    PATTERN = re.compile(r"\bform\s*n-2\b", re.IGNORECASE)

    # Controlled vocabularies
    ALLOWED = {
        "Fund Structure": [
            "Closed End Fund",
            "Business Development Company"
        ],
        "Fund Subtype": [
            "Interval Fund",
            "Tender Offer Fund",
            "Traded CEF",
            "Other Non Traded CEF",
            "Non Traded BDC",
            "Traded BDC"
        ],
        "NAV Reporting Frequency": [
            "Daily",
            "Monthly",
            "Quarterly",
            "Not Applicable"
        ],
        "Suitability": [
            "Accredited Investor",
            "Qualified Client",
            "Qualified Purchaser",
            "No Restriction"
        ]
    }

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, tickers: List[str], text: str) -> dict:
        import openai
        from pathlib import Path

        t = tickers[0]
        fetched = fetch_for_ticker(t, count=1, form_types=("N-2",))
        if not fetched:
            raise HTTPException(404, f"No Form N-2 found for {t}.")
        build_faiss_index_for_ticker(t, reset=True)

        # Grab the raw file
        data_dir = Path(__file__).parent.parent / "ingestion" / "data" / t
        raw_file = next(data_dir.glob("*.xml"), None) or next(data_dir.glob("*.htm*"), None)
        if not raw_file:
            raise HTTPException(500, f"No downloaded Form N-2 for {t}.")
        raw = raw_file.read_text(encoding="utf-8", errors="ignore")

        # Few‐shot prompt
        example = """
            Example:
            Excerpt: "This closed end fund is a Traded CEF. NAV is calculated Daily. Only Accredited Investors may purchase."
            JSON:
            {"Fund Structure":"Closed End Fund","Fund Subtype":"Traded CEF","NAV Reporting Frequency":"Daily","Suitability":"Accredited Investor"}
            """
        prompt = f"""
            You are a financial data extractor. I will give you excerpts from a fund’s Form N-2. You must output EXACTLY one JSON object with these four keys (and only these keys): 
            • Fund Structure  (must be "Closed End Fund" or "Business Development Company")
            • Fund Subtype    (one of "Interval Fund","Tender Offer Fund","Traded CEF","Other Non Traded CEF","Non Traded BDC","Traded BDC")
            • NAV Reporting Frequency ("Daily","Monthly","Quarterly","Not Applicable")
            • Suitability     ("Accredited Investor","Qualified Client","Qualified Purchaser","No Restriction")

            {example}

            Now, here are the excerpts:
            \"\"\"
            {raw[:2000]}  # first 2000 characters
            \"\"\"

            JSON:
            """
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You extract structured data exactly as JSON."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        # Parse the response
        content = resp.choices[0].message.content.strip()
        # strip code fences if any
        content = re.sub(r"```json(.*?)```", r"\1", content, flags=re.S).strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(500, detail=f"Failed to parse JSON from GPT:\n{content}")
        
        if not parsed.get("NAV Reporting Frequency") or parsed["NAV Reporting Frequency"] == "Unknown":
        # look for any of the four allowed keywords
            nav_match = re.search(
                r"\b(Daily|Monthly|Quarterly|Not Applicable)\b",
                raw,
                flags=re.IGNORECASE
            )
            if nav_match:
                # normalize casing to Title Case
                parsed["NAV Reporting Frequency"] = nav_match.group(1).title()

        # 2) Ensure it’s never empty
        if not parsed.get("NAV Reporting Frequency"):
            parsed["NAV Reporting Frequency"] = "Unknown"

        # Final sanity‐check: ensure all keys present
        for key in self.ALLOWED:
            if key not in parsed:
                parsed[key] = "Unknown"
        return {"answer": parsed}


class RAGFallbackHandler(AskHandler):
    def can_handle(self, text: str) -> bool:
        return True

    def handle(self, tickers: List[str], text: str) -> dict:
        results = {}
        for t in tickers:
            ans = answer_question_for_ticker(t, text)
            results[t] = ans
        return {"answer": results}


ASK_HANDLERS: List[AskHandler] = [
    NetIncomeYearHandler(),
    LatestNetIncomeHandler(),
    YearPerformanceHandler(),
    LatestRevenueHandler(),
    ProfitCompareHandler(),
    ProfitPctCompareHandler(),
    StockNewsHandler(),
    FormN2Handler(),
    RAGFallbackHandler(),
]
