import re
from typing import List
from fastapi import HTTPException
import openai
from ingestion.edgar_fetch import fetch_for_ticker

from summarization.extract_metrics import (
    get_latest_revenue,
    get_revenue_by_year,
    get_latest_net_income,
    get_net_income_by_year,
    get_net_income_by_years,
    get_profit_percentage_by_years,
)
from summarization.news_index import retrieve_news_context
from summarization.summarize import answer_question, build_faiss_index_for_ticker, retrieve_context_for_ticker


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
        answers = []
        for t in tickers:
            ni = get_net_income_by_year(t, year)
            if ni:
                answers.append(f"{t} net income {year}: ${ni}")
        if not answers:
            return {"answer": f"No net income data found for {', '.join(tickers)} in {year}."}
        return {"answer": "\n".join(answers)}


class LatestNetIncomeHandler(AskHandler):
    KEYWORDS = ["latest net income", "net income"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return any(kw in txt for kw in self.KEYWORDS) and not re.search(r"20\d{2}", text)

    def handle(self, tickers: List[str], text: str) -> dict:
        answers = []
        for t in tickers:
            ni = get_latest_net_income(t)
            if ni:
                answers.append(f"{t} latest net income: ${ni}")
        if not answers:
            raise HTTPException(status_code=400, detail="No net income data available.")
        return {"answer": "\n".join(answers)}


class RevenueYearHandler(AskHandler):
    PATTERN = re.compile(r"\brevenue\D+?(20\d{2})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, tickers: List[str], text: str) -> dict:
        year = self.PATTERN.search(text).group(1)
        answers = []
        for t in tickers:
            rev = get_revenue_by_year(t, year)
            if rev:
                answers.append(f"{t} revenue {year}: ${rev}")
        if not answers:
            return {"answer": f"No revenue data found for {', '.join(tickers)} in {year}."}
        return {"answer": "\n".join(answers)}


class LatestRevenueHandler(AskHandler):
    KEYWORDS = ["latest revenue", "recent revenue", "revenue"]

    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return any(kw in txt for kw in self.KEYWORDS) and not re.search(r"20\d{2}", text)

    def handle(self, tickers: List[str], text: str) -> dict:
        answers = []
        for t in tickers:
            rev = get_latest_revenue(t)
            if rev:
                answers.append(f"{t} latest revenue: ${rev}")
        if not answers:
            raise HTTPException(status_code=400, detail="No revenue data available.")
        return {"answer": "\n".join(answers)}


class ProfitCompareHandler(AskHandler):
    PATTERN = re.compile(r"\bprofits?\D+?than\D+?([A-Za-z]{2,5})\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return "profit" in text.lower() and "than" in text.lower()

    def handle(self, tickers: List[str], text: str) -> dict:
        if len(tickers) < 2:
            raise HTTPException(status_code=400, detail="Please mention two companies to compare profits.")
        t1, t2 = tickers[0], tickers[1]
        comp = get_net_income_by_years(t1, t2)
        for year in sorted(comp.keys(), reverse=True):
            ni1, ni2 = comp[year]
            try:
                if int(ni1) > int(ni2):
                    return {"answer": f"{t1} had higher net income than {t2} in {year}: ${ni1} vs ${ni2}"}
            except:
                continue
        return {"answer": f"No year found where {t1} net income > {t2}."}


class ProfitPctCompareHandler(AskHandler):
    def can_handle(self, text: str) -> bool:
        return "profit percentage" in text.lower() and "than" in text.lower()

    def handle(self, tickers: List[str], text: str) -> dict:
        if len(tickers) < 2:
            raise HTTPException(status_code=400, detail="Please mention two companies to compare profit percentages.")
        t1, t2 = tickers[0], tickers[1]
        pct_map = get_profit_percentage_by_years(t1, t2)
        for year in sorted(pct_map.keys(), reverse=True):
            pct1, pct2 = pct_map[year]
            if pct2 > pct1:
                return {"answer": f"{t2} had higher profit percentage than {t1} in {year}: {pct2}% vs {pct1}%"}
        return {"answer": f"No year found where {t2} profit percentage > {t1}."}


class StockNewsHandler(AskHandler):
    KEYWORDS = ["stock", "stocks", "news"]

    def can_handle(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.KEYWORDS)

    def handle(self, tickers: List[str], text: str) -> dict:
        answers = []
        for t in tickers:
            latest_rev = get_latest_revenue(t)
            try:
                news = retrieve_news_context(t, text, top_k=5)
            except Exception:
                news = []
            joined = "\n\n---\n\n".join(news)
            prompt = (
                f"You are an equity research assistant. {t}’s most recent revenue was ${latest_rev}.\n\n"
                f"Recent news:\n{joined}\n\nQuestion: {text}"
            )
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            answers.append(f"{t} analysis:\n" + resp.choices[0].message.content.strip())
        return {"answer": "\n\n".join(answers)}

class FormN2Handler(AskHandler):
    """
    Triggered when the question mentions Form N-2.
    Fetches the N-2 filing, retrieves its text chunks, then asks GPT to extract:
      • Fund Structure
      • Fund Subtype
      • NAV Reporting Frequency
      • Suitability
    """
    PATTERN = re.compile(r"\bform\s*n-2\b", re.IGNORECASE)

    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))

    def handle(self, tickers: List[str], text: str) -> dict:
        answers = []
        for t in tickers:
            # 1) ingest only N-2
            fetch_for_ticker(t, count=1, form_types=("N-2",))
            build_faiss_index_for_ticker(t, reset=True)

            # 2) retrieve the top chunks for that Form N-2
            ctx = retrieve_context_for_ticker(t, text, top_k=5)
            joined = "\n\n---\n\n".join(ctx)

            # 3) ask GPT to extract the four fields as JSON
            prompt = f"""
                You are a financial-data extractor. A fund’s Form N-2 contains these fields:
                • Fund Structure
                • Fund Subtype
                • NAV Reporting Frequency
                • Suitability

                Given the following excerpts from {t}’s Form N-2, extract those four fields and return a JSON object:
                \"\"\"{joined}\"\"\"
            """
            # Call OpenAI API to extract the fields
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You output exactly a JSON object with the requested keys."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            # Expect resp.choices[0].message.content to be something like:
            # {"Fund Structure":"Closed-End Fund", "Fund Subtype":"Traded CEF", ...}
            answers.append(f"{t}: {resp.choices[0].message.content.strip()}")
        return {"answer":"\n\n".join(answers)}

class RAGFallbackHandler(AskHandler):
    def can_handle(self, text: str) -> bool:
        return True

    def handle(self, tickers: List[str], text: str) -> dict:
        answers = []
        for t in tickers:
            ans = answer_question(t, text)
            answers.append(f"{t}: {ans}")
        return {"answer": "\n\n".join(answers)}


# Ordered list of handlers; more specific first
ASK_HANDLERS: List[AskHandler] = [
    NetIncomeYearHandler(),
    LatestNetIncomeHandler(),
    RevenueYearHandler(),
    LatestRevenueHandler(),
    ProfitCompareHandler(),
    ProfitPctCompareHandler(),
    StockNewsHandler(),
    FormN2Handler(),
    RAGFallbackHandler(),
]
