import os
import pytest
from fastapi.testclient import TestClient
from typing import List

# Import the FastAPI app
from api.main import app
import ingestion.edgar_fetch as ef

client = TestClient(app)

@pytest.fixture(autouse=True)
def stub_fetch_and_env(monkeypatch, tmp_path):
    # … existing monkeypatches …

    # Stub out the actual S3 upload so we never call boto3 here
    monkeypatch.setattr(ef, '_upload_to_s3', lambda *args, **kwargs: None)

    # Stub fetch_for_ticker itself
    dummy_index = tmp_path / 'AAPL-0001-index.json'
    dummy_index.write_text('{}')
    dummy_xml = tmp_path / 'AAPL-0001.xml'
    dummy_xml.write_text('<xbrl/>')
    monkeypatch.setattr(
        ef, 'fetch_for_ticker',
        lambda ticker, count, form_types: [dummy_index, dummy_xml]
    )
    yield


def test_api_ingest_returns_s3_keys():
    payload = {
        'ticker': 'AAPL',
        'count': 1,
        'form_types': ['10-K']
    }
    response = client.post('/ingest', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'ingested_s3_keys' in data
    keys: List[str] = data['ingested_s3_keys']
    # We stubbed two files, so expect two keys
    assert len(keys) == 2
    # Validate formatting of keys
    assert keys[0].startswith('edgar/AAPL/0001/')
    assert keys[1].startswith('edgar/AAPL/0001/')
