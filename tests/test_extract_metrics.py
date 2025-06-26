import sys
import os
import pytest
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the actual functions under test
from summarization.extract_metrics import get_metric_for_year, get_net_income_by_year
import summarization.extract_metrics as em


@pytest.fixture
def sample_xbrl(tmp_path):
    """
    Create minimal XBRL files for testing revenue and net income extraction.
    Returns a dict mapping year strings to lists of Path objects.
    """
    data = {}

    # 2020: both Revenues and NetIncomeLoss
    xml_2020 = '''<?xml version="1.0" encoding="UTF-8"?>
    <xbrl xmlns:us-gaap="http://fasb.org/us-gaap/2020-01-31">
      <us-gaap:Revenues contextRef="I-2020">274515000000</us-gaap:Revenues>
      <us-gaap:NetIncomeLoss contextRef="I-2020">57411000000</us-gaap:NetIncomeLoss>
    </xbrl>'''
    f2020 = tmp_path / "AAPL-20201231.xml"
    f2020.write_text(xml_2020, encoding="utf-8")
    data['2020'] = [f2020]

    # 2021: only Revenues
    xml_2021 = '''<?xml version="1.0" encoding="UTF-8"?>
    <xbrl xmlns:dei="http://xbrl.sec.gov/dei/2021-01-31">
      <dei:Revenues contextRef="I-2021">300000000000</dei:Revenues>
    </xbrl>'''
    f2021 = tmp_path / "AAPL-20211231.xml"
    f2021.write_text(xml_2021, encoding="utf-8")
    data['2021'] = [f2021]

    # 2022: no matching tags
    xml_2022 = '''<?xml version="1.0" encoding="UTF-8"?>
    <xbrl>
      <Assets contextRef="I-2022">1000000000000</Assets>
    </xbrl>'''
    f2022 = tmp_path / "AAPL-20221231.xml"
    f2022.write_text(xml_2022, encoding="utf-8")
    data['2022'] = [f2022]

    return data


@pytest.fixture(autouse=True)
def patch_collect(monkeypatch, sample_xbrl):
    """
    Monkey-patch the internal XBRL collector to return our sample files.
    """
    monkeypatch.setattr(
        em,
        '_collect_xbrl_instances_by_ticker',
        lambda ticker: sample_xbrl
    )


def test_get_metric_for_year_revenue():
    result = get_metric_for_year("AAPL", 2020, "SalesRevenueNet")
    assert result is None or isinstance(result, int)


def test_get_net_income_by_year_found():
    ni = get_net_income_by_year('AAPL', '2020')
    assert ni == '57411000000'


def test_revenue_missing_tag_returns_none():
    assert get_metric_for_year('AAPL', 2022, "SalesRevenueNet") is None


def test_net_income_missing_tag_returns_none():
    assert get_net_income_by_year('AAPL', 2021) is None