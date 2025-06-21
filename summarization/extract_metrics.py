import re, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import logging
import json

DOCS_DIR = Path(__file__).parent.parent / "ingestion" / "data"

METRIC_TAGS = {
  "revenue":           ["SalesRevenueNet", "Revenues"],
  "net income":        ["NetIncomeLoss"],
  "eps":               ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
  "cash":              ["CashAndCashEquivalentsAtCarryingValue"],
  "assets":            ["Assets"],
  "gross profit":      ["GrossProfit"],
  "dividends":         ["CashDividendsDeclared"],
  # …add more as you go…
}

logger = logging.getLogger(__name__)

def _collect_xbrl_by_form(ticker: str) -> dict[str, dict[str, Path]]:
    """
    Returns something like:
      {
        "10-K": {"2020": Path(...aapl-20200926.xml), ...},
        "10-Q": {"2020": Path(...aapl-20200327.xml), ...}
      }
    """
    base = DOCS_DIR / ticker.upper()
    by_form: dict[str, dict[str, Path]] = {}
    for idx_path in base.glob(f"{ticker.upper()}-*-index.json"):
        meta = json.loads(idx_path.read_text())
        form = meta.get("form")               # ensure you saved `form` in your index JSON
        accession = meta.get("accession")     # e.g. "20200926"
        xml_path = idx_path.with_suffix(".xml")
        if form and xml_path.exists():
            year = accession[:4]
            by_form.setdefault(form, {})[year] = xml_path
    return by_form

def _collect_xbrl_instances_by_ticker(ticker: str) -> Dict[str, Path]:
    ticker = ticker.lower()
    candidates = []
    for file in DOCS_DIR.glob(f"{ticker}-*.xml"):
        # … your existing pattern matching …
        candidates.append(file)

    # build a map year → list of files
    by_year: Dict[str, List[Path]] = {}
    for path in candidates:
        date_match = re.search(r"(\d{8})", path.name)
        if not date_match:
            continue
        y = date_match.group(1)[:4]
        by_year.setdefault(y, []).append(path)

    logger.debug(f"[_collect] {ticker.upper()}: found files by year → "
                  f"{ {year: [p.name for p in paths] for year, paths in by_year.items()} }")
    return by_year

def _pick_latest_for_year(candidates: List[Path]) -> Path:
    # Parse YYYYMMDD and pick max
    def key(p): return int(re.search(r"(\d{8})", p.name).group(1))
    return max(candidates, key=key)

def get_metric_for_year(
    ticker: str,
    year: Union[int, str],
    tag: str
) -> Optional[int]:
    year_str = str(year)
    logger.info(f"get_metric_for_year: {ticker=} {year_str=} {tag=}")

    # collect by form
    by_form = _collect_xbrl_by_form(ticker)

    # try the annual report first
    xml_file = by_form.get("10-K", {}).get(year_str)
    if not xml_file:
        # fallback to Q4 if no 10-K available
        xml_file = by_form.get("10-Q", {}).get(year_str)
    if not xml_file:
        logger.warning(f"No 10-K or Q4 XBRL for {ticker} in {year_str}")
        return None

    logger.debug(f"Using {xml_file.name} for {ticker} {year_str}")
    try:
        tree = ET.parse(str(xml_file))
    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_file.name}: {e}")
        return None

    root = tree.getroot()
    for elem in root.findall(".//*"):
        tagname = elem.tag.split("}", 1)[-1]
        if tagname == tag:
            text = (elem.text or "").strip()
            if text.isdigit():
                return int(text)
    return None

# convenience wrappers


def get_net_income_by_year(ticker: str, year: int) -> Optional[int]:
    return get_metric_for_year(ticker, year, "NetIncomeLoss")
