# File: api/utils.py

import os
import openai
from ingestion.company_map import COMPANY_MAP  # your name→ticker dict

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_tickers_via_gpt(text: str) -> list[str]:
    """
    Ask GPT to pull out all company names mentioned in `text`, map them to tickers.
    Returns a deduplicated list of valid tickers.
    """
    prompt = f"""
        You are a helper that extracts company names from a user question.
        List only the company names mentioned, comma-separated, nothing else.
        Question: {text}
        """
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract company names, comma-separated."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=64
    )
    names = resp.choices[0].message.content
    # split on commas, strip whitespace
    tickers = set()
    for name in map(str.strip, names.split(",")):
        if name in COMPANY_MAP:
            tickers.add(COMPANY_MAP[name])
    return list(tickers)
