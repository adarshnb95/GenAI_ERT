# File: ingestion/edgar_fetch.py
import json, os
import requests
from pathlib import Path
from typing import List, Optional
import boto3

# SEC company tickers JSON for dynamic CIK lookup
CIK_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
XBRL_EXCLUDE = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", "_htm.xml")

HEADERS = {
    "User-Agent": "Your Name your.email@example.com",
    "Accept": "application/json"
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

    print(f"[edgar_fetch] Fetched {len(saved)} filings for {ticker}")
    return saved