import re, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

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

def _collect_xbrl_instances_by_ticker(ticker: str) -> Dict[str, List[Path]]:
    pattern = re.compile(rf"^{re.escape(ticker.lower())}-(\d{{8}})\.xml$")
    by_year: Dict[str, List[Path]] = {}
    for file in DOCS_DIR.glob(f"{ticker.lower()}-*.xml"):
        m = pattern.match(file.name)
        if not m: continue
        year = m.group(1)[:4]
        by_year.setdefault(year, []).append(file)
    return by_year

def _pick_latest_for_year(candidates: List[Path]) -> Path:
    # Parse YYYYMMDD and pick max
    def key(p): return int(re.search(r"(\d{8})", p.name).group(1))
    return max(candidates, key=key)

def get_metric_for_year(ticker: str, year: int, tag: str) -> Optional[int]:
    year_str = str(year)
    instances = _collect_xbrl_instances_by_ticker(ticker).get(year_str, [])
    if not instances: return None
    xml_file = _pick_latest_for_year(instances)
    root = ET.parse(xml_file).getroot()
    for el in root.findall(f".//*[local-name()='{tag}']"):
        txt = (el.text or "").strip()
        if txt.isdigit(): return int(txt)
    return None

# convenience wrappers
def get_revenue_by_year(ticker: str, year: int) -> Optional[int]:
    return (get_metric_for_year(ticker, year, "SalesRevenueNet")
            or get_metric_for_year(ticker, year, "Revenues"))

def get_net_income_by_year(ticker: str, year: int) -> Optional[int]:
    return get_metric_for_year(ticker, year, "NetIncomeLoss")
