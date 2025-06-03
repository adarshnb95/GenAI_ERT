# File: summarization/extract_metrics.py

import re
import xml.etree.ElementTree as ET
from pathlib import Path

# Same DOCS_DIR as before
DOCS_DIR = Path(__file__).parent.parent / "ingestion" / "data"
FILENAME_PATTERN = re.compile(r"^aapl-(\d{8})\.xml$", re.IGNORECASE)

def _collect_xbrl_by_year() -> dict[str, Path]:
    """
    Scan ingestion/data for all 'aapl-YYYYMMDD.xml' files.
    Return a dict mapping 'YYYY' -> Path to the latest Q4 (year-end) file for that year.
    For example:
       '2022' -> Path(".../aapl-20220930.xml")
       '2023' -> Path(".../aapl-20230930.xml")
       etc.
    """
    files_by_date = []  # list of (YYYYMMDD, Path)
    for file in DOCS_DIR.glob("*.xml"):
        m = FILENAME_PATTERN.match(file.name)
        if not m:
            continue
        date_str = m.group(1)  # e.g. "20220930"
        files_by_date.append((date_str, file))

    # Sort descending by date_str (newest first)
    files_by_date.sort(key=lambda x: x[0], reverse=True)

    # For each file, if its date ends with "0930" (i.e. Q4), map its year->Path
    # If you want to include trailing-quarter numbers, you can adjust logic.
    revenue_by_year: dict[str, Path] = {}
    for date_str, path in files_by_date:
        year = date_str[:4]                # "2022", "2023", ...
        month_day = date_str[4:]           # e.g. "0930"
        if month_day == "0930" and year not in revenue_by_year:
            # Found the year-end (Q4) file for this year
            revenue_by_year[year] = path

    return revenue_by_year

def get_revenue_by_year(year: str) -> str:
    """
    Return the <Revenues> (or <SalesRevenueNet>) tag value for Apple in the given calendar year.
    If no Q4 XBRL exists for that year, returns an empty string.
    """
    revenue_map = _collect_xbrl_by_year()
    xbrl_file = revenue_map.get(year)
    if not xbrl_file:
        return ""

    # Parse the chosen XBRL instance
    tree = ET.parse(str(xbrl_file))
    root = tree.getroot()

    for elem in root.findall(".//*"):
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]  # strip namespace
        if tag in ("Revenues", "SalesRevenueNet"):
            return elem.text or ""
    return ""

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

    return values[0] if values else ""

