# File: run_pipeline.py

import argparse
import subprocess
import sys
from pathlib import Path

from summarization.summarize import build_faiss_index
from summarization.news_index import build_news_index

def main(skip_news: bool):
    project_root = Path(__file__).parent.resolve()

    print("\n▶️  Step 1: Running EDGAR ingestion…")
    try:
        subprocess.run([sys.executable, "-m", "ingestion.edgar_fetch"],
                       check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] EDGAR ingestion failed: {e}")
        sys.exit(1)

    print("\n▶️  Step 2: Building filings FAISS index…")
    try:
        build_faiss_index(reset=True)
    except Exception as e:
        print(f"[ERROR] build_faiss_index() failed: {e}")
        sys.exit(1)

    if not skip_news:
        print("\n▶️  Step 3: Fetching & storing news (if NEWSAPI_KEY is set)…")
        news_script = project_root / "ingestion" / "news_fetch.py"
        if news_script.exists():
            try:
                subprocess.run([sys.executable, str(news_script)],
                               check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] News ingestion failed: {e}")

        print("\n▶️  Step 4: Building news FAISS index…")
        try:
            build_news_index(reset=True)
        except Exception as e:
            print(f"[WARN] build_news_index() failed: {e}")

    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full ingestion → summarization → news‐index pipeline."
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Don’t fetch or index news articles; only do EDGAR ingestion + filings index."
    )
    args = parser.parse_args()
    main(skip_news=args.skip_news)
