# File: ingestion/edgar_fetch.py
import json, os
import re


import requests
from pathlib import Path
from typing import List, Optional, Tuple
import boto3
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)  
logger = logging.getLogger(__name__)

# SEC company tickers JSON for dynamic CIK lookup
CIK_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
XBRL_EXCLUDE = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", "_htm.xml")

HEADERS = {
    "User-Agent": "Adarsh Bhandary (adarshnb95@gmail.com)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate"
}


# Where to store filings
DATA_ROOT = Path(__file__).parent / "data"

# Cache for ticker→CIK mapping
_cik_map = None

_S3 = boto3.client(
    's3',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

def _upload_to_s3(local_path: Path, s3_key: str):
    """Helper: push a local file up to S3 under the given key."""
    bucket = os.getenv("EDGAR_S3_BUCKET")
    if not bucket:
        raise RuntimeError("EDGAR_S3_BUCKET environment variable is not set")
    _S3.upload_file(Filename=str(local_path), Bucket=bucket, Key=s3_key)

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
    return _cik_map.get(ticker.upper())

def download_filing_index(
    cik: str,
    accession: str,
    filename: str,
    dest_dir: Path,
) -> Path:
    """
    Download https://data.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json
    into dest_dir/filename and return that Path.
    """
    # ensure directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    idx_url = f"https://data.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
    resp = requests.get(idx_url, headers=HEADERS)
    resp.raise_for_status()

    idx_path = dest_dir / filename
    idx_path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
    return idx_path

def get_latest_filings(
    cik: str,
    form_types: tuple = ("10-K", "10-Q"),
    count: int = 20
) -> List[dict]:
    """
    Fetch recent filings JSON for a CIK. Returns up to `count` filings matching `form_types`.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    # DEBUG: show the headers you’re about to send

    resp = requests.get(url, headers=HEADERS, timeout=10)

    # DEBUG: confirm what actually went out

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
    Pick the primary XBRL instance (or fallback to HTML/zip), download it, 
    and return the local Path (or None if not found/download fails).
    """
    # 1) Load index once
    with open(index_path, encoding="utf-8") as f:
        items = f.read()
    idx = json.loads(items).get("directory", {}).get("item", [])

    # 2) Pick file in priority order
    xbrls = [
        e["name"] for e in idx
        if e["name"].lower().endswith(".xml")
        and not any(e["name"].lower().endswith(exc) for exc in XBRL_EXCLUDE)
    ]
    htmls = [e["name"] for e in idx if e["name"].lower().endswith((".htm", ".html"))]
    zips  = [e["name"] for e in idx if e["name"].lower().endswith(".xbrl.zip")]

    for choice_list in (xbrls, htmls, zips):
        if choice_list:
            filename = choice_list[0]
            break
    else:
        print(f"[WARN] No XBRL/HTML/ZIP found for CIK={cik}, accession={accession}")
        return None

    # 3) Download
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filename}"
    resp = requests.get(url, headers=HEADERS)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Download failed for {filename}: {e}")
        return None

    out_path = dest_dir / filename
    out_path.write_bytes(resp.content)
    print(f"[edgar_fetch] Downloaded: {filename}")

    # 4) If it's a ZIP, you might want to unzip and return the inner XML
    if filename.lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(out_path, 'r') as z:
            # extract the first .xml inside
            xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                return out_path
            xml_name = xml_names[0]
            z.extract(xml_name, dest_dir)
            return dest_dir / xml_name

    return out_path


def fetch_for_ticker(
    ticker: str,
    count: int = 20,
    form_types: tuple = ("10-K", "10-Q")
) -> List[Path]:
    """
    Fetch filings for `ticker`, save locally *and* push index + docs to S3.
    """
    ticker = ticker.upper()
    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise ValueError(f"Unknown ticker {ticker}")

    out_dir = DATA_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in out_dir.glob(f"{ticker}-*-index.json")}
    saved: List[Path] = []
    filings = get_latest_filings(cik, form_types=form_types, count=count)

    logger.info(f"Fetched {len(filings)} filings. Dates:")

    for f in filings:
        accession = f["accession"]
        idx_name = f"{ticker}-{accession}-index.json"
        idx_path = out_dir / idx_name

        # Skip if we already have it locally
        if idx_name in existing:
            saved.append(idx_path)
            # Even if it's cached locally, push to S3
            s3_index_key = f"edgar/{ticker}/{accession}/{idx_name}"
            _upload_to_s3(idx_path, s3_index_key)
            continue

        # 1) Download index JSON
        idx_url = (
            f"https://data.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
        )
        resp = requests.get(idx_url, headers=HEADERS)
        resp.raise_for_status()
        idx_data = resp.json()

        # 2) Save index locally
        idx_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
        saved.append(idx_path)

        # 3) Upload index JSON to S3
        s3_index_key = f"edgar/{ticker}/{accession}/{idx_path.name}"
        _upload_to_s3(idx_path, s3_index_key)

        # 4) Download primary component (XML/HTML/etc.)
        comp = choose_and_download(cik, accession, str(idx_path), dest_dir=out_dir)
        if comp:
            comp_path = Path(comp)
            saved.append(comp_path)

            # 5) Upload component to S3
            s3_comp_key = f"edgar/{ticker}/{accession}/{comp_path.name}"
            _upload_to_s3(comp_path, s3_comp_key)
        
        logger.info(f"{f['filingDate']} - {f['formType']} - {f['accessionNumber']}")

    print(f"[edgar_fetch] Fetched {len(saved)} filings for {ticker}")
    return saved

def fetch_xbrl_for_year(ticker: str, year: int, form_type: str = "10-K") -> Optional[Path]:
    """
    Find the one filing of type form_type in `year` for this ticker,
    download its index.json and XBRL instance, and return the Path to the .xml.
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return None

    # 1) pull its filings list
    filings = get_latest_filings(cik, form_types=(form_type,), count=1000)
    # look for the first one whose 'date' starts with our year
    target = next((f for f in filings if f["date"].startswith(str(year))), None)
    if not target:
        return None

    accession = target["accession"]
    dest_dir = DATA_ROOT / ticker / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 2) download index.json
    idx_filename = f"{ticker}-{accession}-index.json"
    idx_path = download_filing_index(cik, accession, idx_filename, dest_dir)

    # 3) pick & download the XBRL instance
    xml_path = choose_and_download(
        cik,
        accession,
        str(idx_path),
        dest_dir=dest_dir
    )
    return xml_path

# HEADERS = {"User-Agent": "GenAI_ERT"}  # or whatever you’re already using

def download_filing_index(
    cik: str,
    accession: str,
    idx_filename: str,
    dest_dir: Path
) -> Path:
    """
    Download the EDGAR index.json for this (CIK, accession), save it as dest_dir/idx_filename,
    and return that Path. Raises on any HTTP error.
    """
    # EDGAR stores filings under /edgar/data/<integer CIK>/
    # and then a folder named by the accession with dashes removed.
    # e.g. accession "0000320193-20-000096" → folder "000032019320000096"
    clean_acc = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{clean_acc}/{idx_filename}"
    out_path = dest_dir / idx_filename

    resp = requests.get(idx_url, headers=HEADERS)
    resp.raise_for_status()

    out_path.write_text(resp.text, encoding="utf-8")
    return out_path



def fetch_financial_fact(cik: str, fact_name: str, year: int, period: str = "FY") -> float:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    facts = resp.json()["facts"]["us-gaap"][fact_name]["units"]["USD"]
    for item in facts:
        if item.get("fp") == period and item.get("fy") == year:
            return item["val"]
    raise KeyError(f"{fact_name} for {period} {year} not found")

# Mapping of metric keys to human-readable labels
FACT_METRICS = {
    "NetIncomeLoss": ["earn", "earnings", "income", "net income", "profit"],
    "Revenues":       ["revenue", "sales", "turnover"],

    # new metrics:
    "OperatingIncomeLoss": [
      "operating income", "operatingprofit"
    ],
    "NetCashProvidedByUsedInOperatingActivities": [
      "cash flow", "cash from operations", "operating cash flow"
    ],
    "EarningsPerShareBasic": [
      "eps", "earnings per share", "basic eps"
    ],
    "Assets": [
      "assets", "total assets"
    ],
    "Liabilities": [
      "liabilities", "total liabilities"
    ],
    "StockholdersEquity": [
      "equity", "shareholders’ equity", "stockholders equity"
    ]
}

def extract_fact_request(question: str):
    q = question.lower()
    # 1) year or relative:
    m_rel = re.search(r"\blast year\b", q)
    if m_rel:
        year = datetime.now().year - 1
    else:
        m_year = re.search(r"\b(19|20)\d{2}\b", q)
        year = int(m_year.group(0)) if m_year else None

    # 2) quarter
    m_q = re.search(r"\bq([1-4])\b", q)
    period = f"Q{m_q.group(1)}" if m_q else "FY"

    # 3) metric
    for tag, keywords in FACT_METRICS.items():
        if any(kw in q for kw in keywords):
            return tag, year, period

    return None, None, None