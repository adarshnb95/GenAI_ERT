import os
import csv
import re
from ingestion.edgar_fetch import TICKER_CIK, get_latest_filings

DATA_DIR = 'ingestion/data'
OUT_CSV  = 'classifier/data/labels.csv'
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 1) Build a map: basename → form_type from get_latest_filings()
form_map = {}
for ticker, cik in TICKER_CIK.items():
    for f in get_latest_filings(cik, form_types=("10-K","10-Q"), count=50):
        # f['filename'] is "<CIK>-<accession>"
        form_map[f['filename']] = f['form']

# 2) Prepare regex to extract from HTML
html_pattern = re.compile(r'CONFORMED SUBMISSION TYPE:\s*(10-[KQ])', re.IGNORECASE)

rows = []
for root, _, files in os.walk(DATA_DIR):
    for fname in sorted(files):
        if not fname.lower().endswith(('.xbrl', '.htm', '.html', '-index.json')):
            continue

        path = os.path.join(root, fname)
        rel  = path.replace('\\','/')
        base = os.path.splitext(fname)[0]  # "<CIK>-<accession>-index" or "<CIK>-<accession>"

        label = ""

        # A) HTML files: scan inside for the header
        if fname.lower().endswith(('.htm','.html')):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    snippet = f.read(8192)
                m = html_pattern.search(snippet)
                if m:
                    label = m.group(1).upper()
            except Exception:
                pass

        # B) Other files (XBRL, index JSON): infer via form_map
        if not label:
            # strip any "-index" suffix
            key = base.replace('-index','')
            label = form_map.get(key, "")

        rows.append([rel, label])

# 3) Write out
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as out:
    w = csv.writer(out)
    w.writerow(['filepath','label'])
    w.writerows(rows)

print(f"Wrote {len(rows)} entries; auto-labeled {sum(1 for _,l in rows if l)} rows.")
