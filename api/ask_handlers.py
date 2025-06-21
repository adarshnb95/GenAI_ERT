import re
import json
import openai
from pathlib import Path
from fastapi import HTTPException
from typing import List

from ingestion.edgar_fetch import fetch_for_ticker
from summarization.summarize import (
    build_faiss_index_for_ticker,
    answer_question_for_ticker,
)
from summarization.extract_metrics import (
    get_revenue_by_year,
    get_net_income_by_year,
)
from summarization.news_index import retrieve_news_context

class AskHandler:
    def can_handle(self, text: str) -> bool: ...
    def handle(self, tickers: List[str], text: str) -> dict: ...

class SimpleMetricHandler(AskHandler):
    """Handles 'What was X revenue/net income in the year YYYY?'"""
    METRIC_FUNCS = {
        "revenue": get_revenue_by_year,
        "net income": get_net_income_by_year,
    }
    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return "in the year" in txt and any(m in txt for m in self.METRIC_FUNCS)
    def handle(self, tickers, text):
        m = re.search(r"in the year (\d{4})", text)
        if not m:
            raise HTTPException(400, "Year not found.")
        year = int(m.group(1))
        metric = next(m for m in self.METRIC_FUNCS if m in text.lower())
        fn = self.METRIC_FUNCS[metric]
        parts = []
        for t in tickers:
            v = fn(t, year)
            parts.append(f"{t}: ${v:,}" if v else f"{t}: N/A")
        return {"answer": "; ".join(parts)}

class CompareHandler(AskHandler):
    """Handles comparisons like 'When did AAPL have more net income than MSFT?'"""
    def can_handle(self, text: str) -> bool:
        txt = text.lower()
        return any(kw in txt for kw in ["more", "than"]) and "in the year" in txt
    def handle(self, tickers, text):
        if len(tickers) < 2:
            raise HTTPException(400, "Need two tickers to compare.")
        t1, t2 = tickers[:2]
        year = int(re.search(r"in the year (\d{4})", text).group(1))
        ni1 = get_net_income_by_year(t1, year)
        ni2 = get_net_income_by_year(t2, year)
        if ni1 is None or ni2 is None:
            return {"answer": f"Data not available for both in {year}."}
        winner = t1 if ni1 > ni2 else t2
        return {"answer": f"In {year}, {winner} had higher net income (${max(ni1,ni2):,})."}

class NewsHandler(AskHandler):
    """Handles anything with 'news' or 'stock' by doing a RAG over news."""
    def can_handle(self, text: str) -> bool:
        return any(w in text.lower() for w in ("stock", "news"))
    def handle(self, tickers, text):
        out = {}
        for t in tickers:
            snippets = retrieve_news_context(t, text, top_k=5)
            joined = "\n\n---\n\n".join(snippets)
            prompt = (
                f"You are an equity research assistant. {t}'s recent news:\n{joined}\n\nQuestion: {text}"
            )
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You summarize finance news."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            out[t] = resp.choices[0].message.content.strip()
        return {"answer": out}

class FormN2Handler(AskHandler):
    PATTERN = re.compile(r"\bform\s*n-2\b", re.IGNORECASE)
    ALLOWED = {
      "Fund Structure": ["Closed End Fund","Business Development Company"],
      "Fund Subtype": ["Interval Fund","Tender Offer Fund","Traded CEF","Other Non Traded CEF","Non Traded BDC","Traded BDC"],
      "NAV Reporting Frequency": ["Daily","Monthly","Quarterly","Not Applicable"],
      "Suitability": ["Accredited Investor","Qualified Client","Qualified Purchaser","No Restriction"]
    }
    def can_handle(self, text: str) -> bool:
        return bool(self.PATTERN.search(text))
    def handle(self, tickers, text):
        t = tickers[0]
        files = fetch_for_ticker(t, count=1, form_types=("N-2",))
        if not files:
            raise HTTPException(404, f"No Form N-2 for {t}.")
        build_faiss_index_for_ticker(t, reset=True)
        raw = Path(files[0].parent / files[0].with_suffix(".xml").name).read_text(errors="ignore")
        # few-shot prompt omitted for brevity…
        content = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[  # your few-shot
                {"role":"system","content":"Extract exactly JSON."},
                {"role":"user","content": raw[:2000] }
            ],
            temperature=0.0,
            max_tokens=200
        ).choices[0].message.content
        # parse + sanitize…
        parsed = json.loads(re.sub(r"```json|```","", content))
        # fill missing keys with "Unknown"
        for k in self.ALLOWED:
            if k not in parsed or parsed[k] not in self.ALLOWED[k]:
                parsed[k] = "Unknown"
        return {"answer": parsed}

class RAGFallbackHandler(AskHandler):
    def can_handle(self, text): return True
    def handle(self, tickers, text):
        return {"answer": {t: answer_question_for_ticker(t, text) for t in tickers}}

ASK_HANDLERS: List[AskHandler] = [
    SimpleMetricHandler(),
    CompareHandler(),
    NewsHandler(),
    FormN2Handler(),
    RAGFallbackHandler(),
]
