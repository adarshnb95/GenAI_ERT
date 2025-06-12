# File: ingestion/edgar_fetch.py
import json
import requests
from pathlib import Path
from typing import List, Optional

# SEC company tickers JSON for dynamic CIK lookup
CIK_JSON_URL = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": "Your Name your.email@example.com",
    "Accept": "application/json"
}

# Where to store filings
DATA_ROOT = Path(__file__).parent / "data"

# Cache for ticker→CIK mapping
_cik_map = None


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Dynamically fetch and cache the SEC CIK for a given ticker using a public JSON file.
    Returns a zero-padded 10-digit CIK string, or None if not found.
    """
    global _cik_map
    if _cik_map is None:
        resp = requests.get(CIK_JSON_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # data is a dict mapping some key → { "cik_str": "...", "ticker": "...", ... }
        _cik_map = {
            entry["ticker"]: str(entry["cik_str"]).zfill(10)
            for entry in data.values()
        }
    return _cik_map.get(ticker)

def get_latest_filings(
    cik: str,
    form_types: tuple = ("10-K", "10-Q"),
    count: int = 20
) -> List[dict]:
    """
    Fetch recent filings JSON for a CIK. Returns up to `count` filings matching `form_types`.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    recent = resp.json().get("filings", {}).get("recent", {})

    filings = []
    for acc, form, date in zip(
        recent.get("accessionNumber", []),
        recent.get("form", []),
        recent.get("filingDate", []),
    ):
        if form in form_types:
            filings.append({"accession": acc.replace("-", ""), "form": form, "date": date})
            if len(filings) >= count:
                break

    return filings


def choose_and_download(
    cik: str,
    accession: str,
    index_path: str,
    dest_dir: Path
) -> Optional[Path]:
    """
    Given an EDGAR index JSON, pick the primary XBRL instance or HTML and download it.
    Returns the Path to the saved file or None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    items = idx.get("directory", {}).get("item", [])

    instance_name = None
    for entry in items:
        name = entry.get("name", "")
        lower = name.lower()
        if lower.endswith(".xml") and not any(
            lower.endswith(s) for s in ("_cal.xml","_def.xml","_lab.xml","_pre.xml","_htm.xml")
        ) and lower != "filingsummary.xml":
            instance_name = name
            break
    if not instance_name:
        for entry in items:
            name = entry.get("name", "")
            if name.lower().endswith((".htm", ".html")):
                instance_name = name
                break
    if not instance_name:
        print(f"[WARN] No XBRL/HTML found for CIK={cik}, accession={accession}")
        return None

    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{instance_name}"
    resp = requests.get(url, headers=HEADERS)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Download failed for {instance_name}: {e}")
        return None

    out_path = dest_dir / instance_name
    out_path.write_bytes(resp.content)
    print(f"[edgar_fetch] Downloaded: {instance_name}")
    return out_path


def fetch_for_ticker(ticker: str, count: int = 20, form_types: tuple = ("10-K","10-Q")) -> List[Path]:
    """
    Fetch specified filings for `ticker`, saving index JSON and downloading artifacts.
    """
    ticker = ticker.upper()
    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise ValueError(f"Unable to find CIK for ticker {ticker}.")

    out_dir = DATA_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    filings = get_latest_filings(cik, form_types=form_types, count=count)
    for f in filings:
        accession = f["accession"]
        # Download the full EDGAR index.json
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
        resp = requests.get(idx_url, headers=HEADERS)
        resp.raise_for_status()
        idx_data = resp.json()

        idx_path = out_dir / f"{ticker}-{accession}-index.json"
        idx_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
        saved.append(idx_path)

        choose_and_download(cik, accession, str(idx_path), dest_dir=out_dir)

    print(f"[edgar_fetch] Fetched {len(saved)} filings for {ticker} into {out_dir}")
    return saved
