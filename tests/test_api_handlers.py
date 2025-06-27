import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from summarization.extract_metrics import get_metric_for_year  # if needed
from api.ask_handlers import SimpleMetricHandler  # import the correct handler

# Ensure project root is on PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from api.main import app
import ingestion.edgar_fetch as efetch
import summarization.summarize as ssummarize
import summarization.extract_metrics as emetrics

@ pytest.fixture(autouse=True)
def no_remote_ingest(monkeypatch, tmp_path):
    """
    Prevent any actual EDGAR fetch or FAISS build in tests.
    Instead, supply fake local XBRL data and stub out fetch/index.
    """
    # Prepare fake XBRL data for AAPL 2020
    fake_file = tmp_path / 'AAPL-20201227.xml'
    fake_file.write_text(
        '''<?xml version="1.0" encoding="UTF-8"?>
        <xbrl>
          <Revenues>274515000000</Revenues>
          <NetIncomeLoss>57411000000</NetIncomeLoss>
        </xbrl>''', encoding='utf-8')

    # Stub fetch_for_ticker to return the fake file
    def stub_fetch(ticker, **kwargs):
        return [fake_file]
    monkeypatch.setattr(efetch, 'fetch_for_ticker', stub_fetch)

    # Stub FAISS index build to do nothing
    monkeypatch.setattr(ssummarize, 'build_faiss_index_for_ticker', lambda ticker, reset: None)

    # Monkey-patch the XBRL collector to return our fake file for 2020
    monkeypatch.setattr(
        emetrics,
        '_collect_xbrl_instances_by_ticker',
        lambda ticker: {'2020': [fake_file]}
    )
    yield

@ pytest.fixture
def client():
    return TestClient(app)


def test_simple_metric_handler():
    handler = SimpleMetricHandler()

    question = "What was AAPL revenue in the year 2020?"
    tickers = ["AAPL"]

    if handler.can_handle(question):
        result = handler.handle(tickers, question)
        print(result)
        assert "AAPL" in result["answer"]


def test_net_income_handler_direct_parse(client):
    """
    Verify that a net income question is answered via direct XBRL parsing.
    """
    payload = {'text': 'What was AAPL net income in 2020?'}
    response = client.post('/ask', json=payload)
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    data = response.json()
    assert 'answer' in data
    assert 'AAPL' in data['answer']
    assert data['answer']['AAPL'] == '57411000000'
