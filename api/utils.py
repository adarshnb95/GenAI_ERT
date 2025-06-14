# File: api/utils.py
import os
import re
import openai
import requests
from typing import List, Optional

# Configure OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Yahoo Finance search endpoint for dynamic ticker lookup
SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


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
    # 1) Try finding uppercase tokens 2–5 letters that Yahoo recognizes
    raw_candidates = re.findall(r"\b[A-Z]{2,4}\b", text)
    tickers = []
    for tok in raw_candidates:
        # 2) Confirm that Yahoo (or SEC CIK map) agrees it's a real symbol
        resolved = lookup_ticker(tok)
        if resolved and resolved.upper() == tok and tok not in tickers:
            tickers.append(tok)

    if tickers:
        return tickers

    # GPT fallback
    prompt = (
        "List all stock ticker symbols mentioned in this question, comma-separated, "
        "with nothing else. If you see none, say NONE.\n\n"
        f"{text}"
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant that only outputs ticker symbols."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=30
    )
    content = resp.choices[0].message.content.strip()
    # Split on commas, strip, uppercase
    symbols = [s.strip().upper() for s in content.split(",") if s.strip()]
    # *** Filter out any “NONE” or non-alphanumeric tokens ***
    symbols = [s for s in symbols if s.isalpha() and s.upper() != "NONE"]
    return symbols
