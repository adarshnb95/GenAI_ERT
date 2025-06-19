import sys
import os
# Ensure project root is on PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

import json
import pytest
from pathlib import Path
from moto import mock_aws
import boto3

import ingestion.edgar_fetch as ef

# Alias the fetch function for clarity
fetch_for_ticker = ef.fetch_for_ticker

# Auto-start moto and configure S3
@pytest.fixture(autouse=True)
def s3_bucket(monkeypatch):
    mock = mock_aws()
    mock.start()

    # Set AWS env vars
    monkeypatch.setenv('EDGAR_S3_BUCKET', 'test-bucket')
    monkeypatch.setenv('AWS_REGION', 'us-east-1')

    # Recreate the module's S3 client
    new_s3 = boto3.client('s3', region_name='us-east-1')
    monkeypatch.setattr(ef, '_S3', new_s3)
    

    # Create the test bucket
    new_s3.create_bucket(Bucket='test-bucket')

    yield
    mock.stop()

# Stub network and file system
class DummyResponse:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        return None
    def json(self):
        return self._data

@pytest.fixture(autouse=True)
def stub_network_calls(monkeypatch, tmp_path):
    # Stub CIK lookup
    monkeypatch.setattr(ef, 'get_cik_for_ticker', lambda ticker: '0000000001')
    # Stub filings list
    monkeypatch.setattr(
        ef, 'get_latest_filings',
        lambda cik, form_types, count: [
            {'accession': '0001', 'form': '10-K', 'date': '2020-12-31'}
        ]
    )
    # Stub index.json download
    monkeypatch.setattr(
        ef.requests, 'get',
        lambda *args, **kwargs: DummyResponse({'directory': {'item': []}})
    )
    # Prepare a fake XML for the filing
    fake_xml = tmp_path / 'AAPL-0001.xml'
    fake_xml.write_text('<xbrl><Revenues>123</Revenues></xbrl>', encoding='utf-8')
    # Stub component fetch
    monkeypatch.setattr(
        ef, 'choose_and_download',
        lambda cik, accession, idx_path, dest_dir: fake_xml
    )
    yield

# Actual test
def test_s3_ingest_pushes_to_bucket():
    # Perform ingestion
    results = fetch_for_ticker('AAPL', count=1, form_types=('10-K',))
    # Should return the index file path
    assert results and results[0].name == 'AAPL-0001-index.json'

    # Check S3 contents
    resp = ef._S3.list_objects_v2(Bucket='test-bucket')
    keys = {obj['Key'] for obj in resp.get('Contents', [])}

    expected_index = 'edgar/AAPL/0001/AAPL-0001-index.json'
    expected_xml   = 'edgar/AAPL/0001/AAPL-0001.xml'
    assert expected_index in keys or expected_xml in keys
