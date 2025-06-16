# File: summarization/extract_metrics.py

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, Dict, List

# Same DOCS_DIR as before
DOCS_DIR = Path(__file__).parent.parent / "ingestion" / "data"
FILENAME_PATTERN = re.compile(r"^aapl-(\d{8})\.xml$", re.IGNORECASE)

def _collect_xbrl_instances_by_ticker(ticker: str) -> dict[str, Path]:
    """
    Return a dict of { 'YYYY' : Path to the Q4 instance XBRL for that year } for any given ticker.
    Looks for files named like 'TICKER-YYYYMMDD.xml' under ingestion/data.
    """
    ticker = ticker.lower()
    candidates = []  # list of (YYYYMMDD, Path)
    for file in DOCS_DIR.glob(f"{ticker}-*.xml"):
        m = FILENAME_PATTERN.match(file.name)
        if m and m.group(1).lower() == ticker:
            date_str = m.group(2)  # e.g. '20200926'
            candidates.append((date_str, file))

    candidates.sort(key=lambda x: x[0], reverse=True)

    revenue_by_year: dict[str, Path] = {}
    for date_str, path in candidates:
        year = date_str[:4]
        month_day = date_str[4:]
        # We treat Q4 (month_day == '0930' or '0926' etc.) as year-end:
        # EDGAR often uses e.g. 20200926 for fiscal year ending Sep 26, 2020
        if month_day.startswith("09") and year not in revenue_by_year:
            revenue_by_year[year] = path

    return revenue_by_year

def get_revenue_by_year(ticker: str, year: Union[int, str]) -> Optional[str]:
    """
    Extracts the 'Revenues' or 'SalesRevenueNet' line item from the Q4 XBRL for `ticker` in `year`.
    Returns the raw string (e.g. '274515000000'), or None if not found.
    """
    year_str = str(year)
    revenue_map = _collect_xbrl_instances_by_ticker(ticker)
    candidates = revenue_map.get(year_str)
    if not candidates:
        return None

    # Prefer Q4 (Dec 31) filings
    q4_files = [p for p in candidates if '1231' in p.name]
    xml_file = q4_files[0] if q4_files else candidates[0]

    tree = ET.parse(str(xml_file))
    root = tree.getroot()

    # Iterate through all elements and match local tag names
    for elem in root.iter():
        tag = elem.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        if tag in ('Revenues', 'SalesRevenueNet') and elem.text:
            return elem.text.strip()
    return None

def get_latest_net_income() -> str:
    """
    (Existing function) Finds the newest 'aapl-YYYYMMDD.xml' and extracts <NetIncomeLoss>.
    """
    pattern = re.compile(r"^aapl-(\d{8})\.xml$", re.IGNORECASE)
    candidates = []
    for file in DOCS_DIR.glob("*.xml"):
        m = pattern.match(file.name)
        if m:
            candidates.append((m.group(1), file))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = candidates[0][1]

    tree = ET.parse(str(latest_file))
    root = tree.getroot()

    values = []
    for elem in root.findall(".//*"):
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        if tag == "NetIncomeLoss":
            values.append(elem.text or "")
    return values[0] if values else ""


def get_latest_revenue() -> str:
    """
    Finds the newest 'aapl-YYYYMMDD.xml' and extracts <Revenues> or <SalesRevenueNet>.
    Returns the numeric text (e.g. '394328000000') or '' if not found.
    """
    pattern = re.compile(r"^aapl-(\d{8})\.xml$", re.IGNORECASE)
    candidates = []
    for file in DOCS_DIR.glob("*.xml"):
        m = pattern.match(file.name)
        if m:
            candidates.append((m.group(1), file))
    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = candidates[0][1]

    tree = ET.parse(str(latest_file))
    root = tree.getroot()

    # Look for either <Revenues> or <SalesRevenueNet>
    values = []
    for elem in root.findall(".//*"):
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        if tag in ("Revenues", "SalesRevenueNet"):
            values.append(elem.text or "")

    return values[0] if values else None

def get_net_income_by_year(ticker: str, year: Union[int, str]) -> Optional[str]:
    """
    Extracts the 'NetIncomeLoss' line item from the Q4 XBRL for `ticker` in `year`.
    Returns the raw string (e.g. '57411000000'), or None if not found.
    """
    year_str = str(year)
    income_map = _collect_xbrl_instances_by_ticker(ticker)
    candidates = income_map.get(year_str)
    if not candidates:
        return None

    # Prefer Q4 (Dec 31) filings
    q4_files = [p for p in candidates if '1231' in p.name]
    xml_file = q4_files[0] if q4_files else candidates[0]

    tree = ET.parse(str(xml_file))
    root = tree.getroot()

    for elem in root.iter():
        tag = elem.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        if tag == 'NetIncomeLoss' and elem.text:
            return elem.text.strip()
    return None


def get_net_income_by_years(ticker1: str, ticker2: str) -> dict[str, tuple]:
    """
    Return a dict { year: (ni1, ni2) } for each year where both tickers have Q4 XBRL.
    """
    m1 = _collect_xbrl_instances_by_ticker(ticker1)
    m2 = _collect_xbrl_instances_by_ticker(ticker2)
    common_years = set(m1.keys()) & set(m2.keys())
    result = {}
    for year in common_years:
        ni1 = get_net_income_by_year(ticker1, year)
        ni2 = get_net_income_by_year(ticker2, year)
        if ni1 and ni2:
            result[year] = (ni1, ni2)
    return result

def get_profit_percentage_by_years(ticker1: str, ticker2: str) -> dict[str, tuple]:
    """
    Return { year: (pct1, pct2) } where:
      pct = round(100 * net_income / revenue, 2)
    Skip if revenue or net income is missing or zero.
    """
    years_data = get_net_income_by_years(ticker1, ticker2).keys()
    result = {}
    for year in years_data:
        ni1 = get_net_income_by_year(ticker1, year)
        rev1 = get_revenue_by_year(ticker1, year)
        ni2 = get_net_income_by_year(ticker2, year)
        rev2 = get_revenue_by_year(ticker2, year)
        try:
            pct1 = round(100 * float(ni1) / float(rev1), 2)
            pct2 = round(100 * float(ni2) / float(rev2), 2)
            result[year] = (pct1, pct2)
        except:
            continue
    return result