# File: api/ask_handlers.py

import re
from typing import Optional
from fastapi import HTTPException

# Import whichever extract‐metrics or news helpers you need:
from summarization.extract_metrics import (
    get_revenue_by_year,
    get_latest_revenue,
    get_latest_net_income,
    get_net_income_by_year,
    get_net_income_by_years,
    get_profit_percentage_by_years,
)
from summarization.news_index import retrieve_news_context
from summarization.summarize import answer_question
import openai


class AskHandler:
    """
    Base class for all `/ask` handlers. Subclasses must implement:
      - can_handle(text: str) -> bool
      - handle(text: str) -> dict  (returns {"answer": "..."} or raises HTTPException)
    """
    def can_handle(self, text: str) -> bool:
        raise NotImplementedError

    def handle(self, text: str) -> dict:
        raise NotImplementedError


class NetIncomeYearHandler(AskHandler):
    """
    Match questions like:
      “What was AAPL’s net income in 2022?”
    """
    PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?net income\D+?(20\d{2})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, text: str) -> dict:
        m = self.PATTERN.search(text)
        ticker, year = m.group(1).upper(), m.group(2)
        ni = get_net_income_by_year(ticker, year)
        if not ni:
            return {"answer": f"No net income data found for {ticker} in {year}."}
        return {"answer": f"{ticker} net income {year}: ${ni}"}


class LatestNetIncomeHandler(AskHandler):
    """
    Match questions asking “What is the latest net income?” without specifying a ticker+year.
    e.g. “What is Apple’s latest net income?” or simply “net income” if you assume default ticker=AAPL
    """
    KEYWORDS = ["net income", "latest net income"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return any(kw in txt for kw in self.KEYWORDS)

    def handle(self, text: str) -> dict:
        ni_latest = get_latest_net_income()
        if not ni_latest:
            raise HTTPException(status_code=400, detail="No net income data available.")
        return {"answer": f"AAPL latest net income: ${ni_latest}"}


class RevenueYearHandler(AskHandler):
    """
    Match questions like:
      “What was AAPL’s revenue in 2021?”
    """
    PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?revenue\D+?(20\d{2})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, text: str) -> dict:
        m = self.PATTERN.search(text)
        ticker, year = m.group(1).upper(), m.group(2)
        rev = get_revenue_by_year(ticker, year)
        if not rev:
            return {"answer": f"No revenue data found for {ticker} in {year}."}
        return {"answer": f"{ticker} revenue {year}: ${rev}"}


class LatestRevenueHandler(AskHandler):
    """
    Match “latest revenue” or “what is the latest revenue?” without year.
    """
    KEYWORDS = ["latest revenue", "recent revenue", "what is revenue"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        # Only match if “revenue” keyword is present and no year pattern is found
        has_kw = any(kw in txt for kw in self.KEYWORDS)
        no_year = not re.search(r"20\d{2}", text)
        return has_kw and no_year

    def handle(self, text: str) -> dict:
        rev_latest = get_latest_revenue()
        if not rev_latest:
            raise HTTPException(status_code=400, detail="No revenue data available.")
        return {"answer": f"AAPL latest revenue: ${rev_latest}"}


class ProfitCompareHandler(AskHandler):
    """
    Match questions like:
      “When did AAPL have more profits than MSFT?”
    """
    PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?profits?\D+?than\D+?([A-Za-z]{2,5})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, text: str) -> dict:
        m = self.PATTERN.search(text)
        t1, t2 = m.group(1).upper(), m.group(2).upper()
        comp = get_net_income_by_years(t1, t2)  # dict: year -> (ni1, ni2)
        for year in sorted(comp.keys(), reverse=True):
            ni1, ni2 = comp[year]
            try:
                if int(ni1) > int(ni2):
                    return {
                        "answer": f"{t1} had higher net income than {t2} in {year}: ${ni1} vs ${ni2}"
                    }
            except:
                continue
        return {"answer": f"No year found where {t1} net income > {t2} net income in available data."}


class ProfitPctCompareHandler(AskHandler):
    """
    Match questions like:
      “When did MSFT have a higher profit percentage than CRM?”
    """
    PATTERN = re.compile(r"\b([A-Za-z]{2,5})\D+?profit percentage\D+?than\D+?([A-Za-z]{2,5})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, text: str) -> dict:
        m = self.PATTERN.search(text)
        t1, t2 = m.group(1).upper(), m.group(2).upper()
        pct_map = get_profit_percentage_by_years(t1, t2)
        for year in sorted(pct_map.keys(), reverse=True):
            pct1, pct2 = pct_map[year]
            if pct2 > pct1:
                return {
                    "answer": f"{t2} had higher profit percentage than {t1} in {year}: {pct2}% vs {pct1}%"
                }
        return {"answer": f"No year found where {t2} profit percentage > {t1} profit percentage in available data."}


class StockNewsHandler(AskHandler):
    """
    Match forward‐looking / “stocks” or “news” questions, e.g.:
      “Based on AAPL’s recent revenue, will their stocks go higher this year? What does the recent news say?”
    """
    KEYWORDS = ["stock", "stocks", "news"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return any(kw in txt for kw in self.KEYWORDS)

    def handle(self, text: str) -> dict:
        # 1) Grab latest Apple revenue (or you can generalize to detect ticker too)
        latest_rev = get_latest_revenue()

        # 2) Look up top‐5 related news snippets
        try:
            news_snips = retrieve_news_context(text, top_k=5)
        except Exception:
            news_snips = []

        joined_news = "\n\n---\n\n".join(news_snips)

        prompt = f"""
You are an equity research assistant. Apple’s most recent revenue was ${latest_rev}.
Below are a few recent news excerpts relevant to Apple:

{joined_news}

Question: {text}
"""
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return {"answer": resp.choices[0].message.content.strip()}


class RAGFallbackHandler(AskHandler):
    """
    A generic retrieval‐augmented GPT fallback if no other handler applies.
    """
    def can_handle(self, text: str) -> bool:
        return True  # always last in the chain

    def handle(self, text: str) -> dict:
        answer = answer_question(text)
        return {"answer": answer}


# Build a registry (ordered by priority)
ASK_HANDLERS = [
    NetIncomeYearHandler(),
    LatestNetIncomeHandler(),
    RevenueYearHandler(),
    LatestRevenueHandler(),
    ProfitCompareHandler(),
    ProfitPctCompareHandler(),
    StockNewsHandler(),
    RAGFallbackHandler(),  # must be last
]
