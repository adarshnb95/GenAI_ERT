import os
import json
import requests
from datetime import datetime
from pathlib import Path

# SEC requires a custom User-Agent identifying your script
HEADERS = {
    "User-Agent": "Your Name your.email@example.com"
}

# Map tickers to CIKs (10-digit strings)
TICKER_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    # Add more tickers/CIKs as needed
}

# Directory for raw filings
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_submission_index(cik: str) -> dict:
    """
    Fetch the company submission index JSON from SEC EDGAR for a given CIK.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_latest_filings(cik: str, form_types=("10-K", "10-Q"), count=2) -> list:
    """
    Returns the most recent filings of specified types (e.g. "10-K", "10-Q").
    """
    submissions = fetch_submission_index(cik)
    filings = submissions.get("filings", {}).get("recent", {})
    results = []

    # Iterate in reverse chronological order
    for i, form in enumerate(filings.get("form", [])):
        if form in form_types:
            accession = filings["accessionNumber"][i].replace("-", "")
            filing_date = filings["filingDate"][i]
            filename = f"{cik}-{accession}"
            results.append({
                "form": form,
                "accession": accession,
                "date": filing_date,
                "filename": filename
            })
            if len(results) >= count:
                break

    return results


def download_filing_index(cik: str, accession: str, basename: str) -> str:
    """
    Download the directory index JSON for a filing and save locally.
    Returns the path to the saved index file.
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    index_path = DATA_DIR / f"{basename}-index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(resp.json(), f, indent=2)
    print(f"Saved index: {index_path.name}")
    return str(index_path)


def download_filing_component(cik: str, accession: str, component_name: str) -> str:
    """
    Download a specific file for a filing (e.g., XBRL or HTML) and save locally.
    Returns the path to the saved component file.
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{component_name}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    file_path = DATA_DIR / component_name
    with open(file_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved component: {component_name}")
    return str(file_path)


def classify_if_available(text: str) -> str:
    """
    Try to import and run classify_text(). If the model isn't trained yet,
    return a default label "UNCLASSIFIED".
    """
    try:
        from classifier.predict import classify_text
        return classify_text(text)
    except Exception:
        return "UNCLASSIFIED"


def choose_and_download(cik: str, accession: str, index_path: str) -> str:
    """
    Open the index JSON, pick the true XBRL instance (.xml) first
    (the one not ending in _cal.xml, _def.xml, _lab.xml, _pre.xml, or FilingSummary.xml).
    If none found, fall back to the first HTML. Return the local path.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    items = idx.get("directory", {}).get("item", [])

    instance_name = None
    for entry in items:
        name = entry.get("name", "").lower()

        # 1) Must end in ".xml"
        if not name.endswith(".xml"):
            continue

        # 2) Exclude common taxonomy or summary files
        if name.endswith("_cal.xml") or name.endswith("_def.xml") \
           or name.endswith("_lab.xml") or name.endswith("_pre.xml") \
           or name.endswith("_htm.xml") \
           or name == "filingsummary.xml":
            continue

        # If we get here, this is likely the XBRL instance:
        instance_name = entry.get("name")
        break

    # 3) If no valid instance found, pick HTML
    html_name = None
    if instance_name is None:
        for entry in items:
            name = entry.get("name", "").lower()
            if name.endswith((".htm", ".html")):
                html_name = entry.get("name")
                break

    chosen = instance_name if instance_name else html_name
    if chosen is None:
        print(f"[WARN] No XBRL instance or HTML found for accession {accession}")
        return ""

    download_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{chosen}"
    local_path = DATA_DIR / chosen

    resp = requests.get(download_url, headers={"User-Agent": "GenAI_ERT"})
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to download {chosen}: {e}")
        return ""

    local_path.write_bytes(resp.content)
    print(f"Downloaded component: {chosen}")
    return str(local_path)


if __name__ == "__main__":
    # Example: fetch & download the last 2 filings for AAPL
    ticker = "AAPL"
    cik = TICKER_CIK.get(ticker.upper())
    if not cik:
        print(f"Ticker {ticker} not found in TICKER_CIK mapping.")
        exit(1)

    print(f"Processing {ticker}...")
    filings = get_latest_filings(cik, count=60)
    for f in filings:
        basename = f["filename"]
        idx_path = download_filing_index(cik, f["accession"], basename)
        component_path = choose_and_download(cik, f["accession"], idx_path)
        if component_path:
            print(f"Downloaded component to: {component_path}\n")
    print("Ingestion complete.")
