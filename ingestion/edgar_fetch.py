import os
import requests
import json
from datetime import datetime
from classifier.predict import classify_text

# SEC requires a custom User-Agent identifying your script
HEADERS = {
    "User-Agent": "Your Name your.email@example.com"
}

# Map tickers to CIKs; extend as needed (10-digit strings)
TICKER_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    # …etc
}

# Directory for raw filings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


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
    Returns the most recent filings of specified types.
    """
    submissions = fetch_submission_index(cik)
    filings = submissions.get("filings", {}).get("recent", {})
    results = []
    for i, form in enumerate(filings.get("form", [])):
        if form in form_types:
            accession = filings["accessionNumber"][i].replace("-", "")
            filing_date = filings["filingDate"][i]
            filename = f"{cik}-{accession}"
            results.append({"form": form, "accession": accession, "date": filing_date, "filename": filename})
            if len(results) >= count:
                break
    return results


def download_filing_index(cik: str, accession: str, basename: str) -> str:
    """
    Download the directory index JSON for a filing and save locally.
    Returns path to the saved index file.
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    index_path = os.path.join(DATA_DIR, f"{basename}-index.json")
    with open(index_path, 'w') as f:
        json.dump(resp.json(), f, indent=2)
    print(f"Saved index: {os.path.basename(index_path)}")
    return index_path


def download_filing_component(cik: str, accession: str, component_name: str) -> None:
    """
    Download a specific file for a filing (e.g., XBRL or HTML) and save locally.
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{component_name}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    file_path = os.path.join(DATA_DIR, component_name)
    with open(file_path, 'wb') as f:
        f.write(resp.content)
    print(f"Saved component: {component_name}")


def choose_and_download(cik: str, accession: str, index_path: str) -> str:
    """
    Parse the index JSON to find the primary XBRL or HTML filing,
    download it, and return its local filepath.
    """
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    # Try XBRL first
    for entry in index_data.get('directory', {}).get('item', []):
        name = entry.get('name', '')
        if name.lower().endswith('.xbrl'):
            download_filing_component(cik, accession, name)
            break
    else:
        # Fallback to HTML
        for entry in index_data.get('directory', {}).get('item', []):
            name = entry.get('name', '')
            if name.lower().endswith(('.htm', '.html')):
                download_filing_component(cik, accession, name)
                break
        else:
            print(f"No XBRL/HTML component found for {accession}")
            return ''

    # Now that it’s downloaded, build the full local path
    local_path = os.path.join(DATA_DIR, name)
    return local_path


if __name__ == "__main__":
    # Example: fetch & download the last 2 filings for AAPL
    for ticker, cik in TICKER_CIK.items():
        print(f"\nProcessing {ticker}...")
        filings = get_latest_filings(cik, count=10)
        for f in filings:
            basename = f['filename']
            idx_path = download_filing_index(cik, f['accession'], basename)
            component_path = choose_and_download(cik, f['accession'], idx_path)
            print(f"Downloaded component to: {component_path}")
            # If you want to classify it immediately:
            # text = open(component_path, 'r').read(2000)
            # label = classify_text(text)
            # print("Classified as", label)

    print("Ingestion complete.")
