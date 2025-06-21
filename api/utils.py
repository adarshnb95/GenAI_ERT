# File: api/utils.py
import os
import re
import openai
import requests
from typing import List, Optional, Dict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configure OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Yahoo Finance search endpoint for dynamic ticker lookup
SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

COMPANY_MAP_PATH = Path(__file__).parent.parent / "company_map.json"

def load_company_map() -> Dict[str, str]:
    if COMPANY_MAP_PATH.exists():
        with open(COMPANY_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_company_map(m: Dict[str, str]) -> None:
    with open(COMPANY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

def add_company_mapping(name: str, ticker: str) -> None:
    """Register a new company-name→ticker mapping."""
    m = load_company_map()
    m[name] = ticker
    save_company_map(m)

def lookup_ticker(company_name: str) -> Optional[str]:
    """
    Query Yahoo Finance to find the primary ticker symbol for a company name.
    Returns the first symbol found, or None if not available.
    """
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("quotes", [])
        if quotes:
            return quotes[0].get("symbol")
    except Exception:
        pass
    return None


def extract_company_names(text: str) -> List[str]:
    """
    Use GPT to extract a list of company names from the user query.
    Returns a list of names, stripped and de-duplicated.
    """
    prompt = (
        f"Extract all company names mentioned in the following query, comma-separated: '{text}'"
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts company names from text."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=64
    )
    content = resp.choices[0].message.content
    # Split on commas
    names = [n.strip() for n in content.split(",") if n.strip()]
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower()); uniq.append(n)
    return uniq


def extract_tickers_from_text(text: str) -> List[str]:
    m = load_company_map()
    tickers = []

    # 1) name‐based lookup (handles “Apple” or “Apple’s”)
    for name, symbol in m.items():
        if re.search(rf"\b{name}(?:'s)?\b", text, flags=re.IGNORECASE):
            tickers.append(symbol)

    # 2) uppercase token check (AAPL, MSFT…)
    for tok in re.findall(r"\b[A-Z]{{2,5}}\b", text):
        if tok not in tickers and lookup_ticker(tok):
            tickers.append(tok)

    if tickers:
        logger.debug(f"[tickers] via rules → {tickers} for text {text!r}")
        return tickers

    # 3) GPT fallback
    prompt = (
        "List all stock ticker symbols mentioned in this question, comma-separated, "
        "with nothing else. If none, respond NONE.\n\n"
        f"{text}"
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You only output ticker symbols."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    symbols = [s.strip().upper() for s in resp.choices[0].message.content.split(",")]
    logger.debug(f"[tickers] via GPT → {symbols} for text {text!r}")

    # **If** GPT gave us a ticker, but we also detect a *new* company name in the text, record it:
    if symbols and symbols != ["NONE"]:
        # cheap heuristic: grab the first capitalized word sequence
        name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        if name_match:
            candidate_name = name_match.group(1)
            # only save if truly new
            if candidate_name not in m:
                add_company_mapping(candidate_name, symbols[0])

    return [s for s in symbols if s != "NONE"]
