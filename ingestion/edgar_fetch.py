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
CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "CRM":  "0000774607",
    # add more tickers here…
}

# Directory for raw filings
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

INGESTION_ROOT = Path(__file__).parent.resolve()
DATA_ROOT      = INGESTION_ROOT / "data"

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


def choose_and_download(
    cik: str,
    accession: str,
    index_path: str,
    dest_dir: Path
) -> Path:
    """
    Given a SEC filing index JSON at `index_path`, pick the primary XBRL instance
    (or fallback to HTML), download it from EDGAR, save under `dest_dir/`, and
    return the Path to the saved file.
    """
    # ensure destination folder exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # load the index JSON
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    items = idx.get("directory", {}).get("item", [])

    # 1) Try to find the true XBRL instance
    instance_name = None
    for entry in items:
        name = entry.get("name", "")
        lower = name.lower()
        if not lower.endswith(".xml"):
            continue
        # exclude taxonomy/support files
        if any(lower.endswith(suffix) for suffix in (
            "_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", "_htm.xml"
        )) or lower == "filingsummary.xml":
            continue
        instance_name = name
        break

    # 2) If no XBRL found, fallback to the first HTML
    chosen = instance_name
    if chosen is None:
        for entry in items:
            name = entry.get("name", "")
            if name.lower().endswith((".htm", ".html")):
                chosen = name
                break

    if not chosen:
        print(f"[WARN] No XBRL or HTML found for CIK={cik}, accession={accession}")
        return None

    # 3) Download the chosen file
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{chosen}"
    headers = {"User-Agent": "GenAI_ERT adisi@example.com"}  # replace with your info
    resp = requests.get(url, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Download failed for {chosen}: {e}")
        return None

    # 4) Write to dest_dir
    out_path = dest_dir / chosen
    out_path.write_bytes(resp.content)
    print(f"[edgar_fetch] Downloaded component: {chosen} → {out_path.name}")

    return out_path

def fetch_for_ticker(ticker: str, count: int = 20) -> list[Path]:
    ticker = ticker.upper()
    if ticker not in CIK_MAP:
        raise ValueError(f"Unknown ticker {ticker}. Add it to CIK_MAP.")
    cik = CIK_MAP[ticker]
    out_dir = DATA_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Get the small metadata listing of filings
    filings = get_latest_filings(cik, count=count)

    saved_paths: list[Path] = []
    for f in filings:
        accession = f["accession"]
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"

        # 2) Download the EDGAR index.json (which has the 'directory' listing)
        idx_url  = f"{base}/index.json"
        resp = requests.get(idx_url, headers=HEADERS)
        resp.raise_for_status()
        idx_json = resp.json()

        # 3) Save that full index.json to disk
        idx_path = out_dir / f"{ticker}-{accession}-index.json"
        idx_path.write_text(json.dumps(idx_json, indent=2), encoding="utf-8")
        saved_paths.append(idx_path)

        # 4) Now that we have a real index.json, pick & download the XBRL/HTML
        choose_and_download(cik, accession, str(idx_path), dest_dir=out_dir)

    print(f"[edgar_fetch] Fetched {len(saved_paths)} filings for {ticker} into {out_dir}")
    return saved_paths

if __name__ == "__main__":
    # Example: fetch & download the last 2 filings for AAPL
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch recent EDGAR filings (10-K / 10-Q) for a given ticker"
    )
    parser.add_argument(
        "ticker",
        help="Stock ticker symbol, e.g. AAPL, MSFT, CRM"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Maximum number of filings to fetch (default: 20)"
    )
    args = parser.parse_args()

    try:
        paths = fetch_for_ticker(args.ticker, count=args.count)
        print(f"✅ Saved {len(paths)} index files under ingestion/data/{args.ticker.upper()}/")
    except Exception as e:
        print(f"❌ Error fetching filings for {args.ticker.upper()}: {e}")
