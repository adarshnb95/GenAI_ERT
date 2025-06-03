# File: scripts/inspect_index.py

import json
from pathlib import Path

# Always locate the project root by going up two levels from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR    = PROJECT_ROOT / "ingestion" / "data"

# Replace this filename with the exact name you saw under ingestion/data/
FILENAME = "0000320193-000032019317000070-index.json"  

index_path = INDEX_DIR / FILENAME

# Debug: print out where we’re looking
print(f"[DEBUG] Looking for index at: {index_path}")

# Now verify the file truly exists
if not index_path.exists():
    print(f"[ERROR] Cannot find {index_path}. Please verify the filename and that ingestion/data is populated.")
    exit(1)

# Read and parse
data_bytes = index_path.read_bytes()
if b'\x00' in data_bytes:
    print(f"[ERROR] {FILENAME} contains null-bytes; likely not valid JSON.")
    exit(1)

text = data_bytes.decode("utf-8", errors="strict")
data = json.loads(text)

items = data.get("directory", {}).get("item", [])
print(f"Found {len(items)} items in {FILENAME}:\n")
for entry in items:
    name = entry.get("name")
    typ  = entry.get("type", "").lower()
    print(f"• name: {name:<60}  |  type: {typ}")
