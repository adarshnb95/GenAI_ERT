import os
import json
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# 1) Paths
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
NEWS_DATA_DIR    = Path(__file__).parent.parent / "ingestion" / "news_data"
NEWS_VECTOR_DIR  = Path(__file__).parent / "news_faiss"
NEWS_META_PATH   = NEWS_VECTOR_DIR / "news_faiss.meta.json"
NEWS_INDEX_PATH  = NEWS_VECTOR_DIR / "news_faiss.index"

# 2) Initialize embedder
embedder = SentenceTransformer(EMBED_MODEL)

def build_news_index(reset: bool = False):
    """
    Reads all news JSON files, embeds their title+description, builds a FAISS index,
    and writes out the index and a metadata JSON mapping vector->(file, title).
    """
    NEWS_VECTOR_DIR.mkdir(exist_ok=True)

    if reset and NEWS_INDEX_PATH.exists():
        NEWS_INDEX_PATH.unlink()

    texts: List[str] = []
    metadata: List[dict] = []

    # 3) Collect all news files
    json_files = sorted(NEWS_DATA_DIR.glob("*.json"))
    if not json_files:
        print("No news files found under", NEWS_DATA_DIR)
        return

    for jf in json_files:
        rec = json.loads(jf.read_text(encoding="utf-8"))
        title = rec.get("title", "")
        desc = rec.get("description", "")
        combined = f"{title} — {desc}".strip()
        if not combined:
            continue

        texts.append(combined)
        metadata.append({
            "file": jf.name,
            "title": title,
            "url": rec.get("url", ""),
            "publishedAt": rec.get("publishedAt", "")
        })

    if not texts:
        print("No non-empty news texts to index.")
        return

    # 4) Embed all news texts
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = vectors.shape[1]

    # 5) Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(NEWS_INDEX_PATH))

    # 6) Write metadata JSON
    with open(NEWS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built news FAISS index with {len(texts)} articles.")

def retrieve_news_context(query: str, top_k: int = 5) -> List[str]:
    """
    Given a user query (e.g. "Apple revenue outlook"), embed it and return the top_k
    news snippets (title + URL + snippet) that are most similar.
    """
    if not NEWS_INDEX_PATH.exists() or not NEWS_META_PATH.exists():
        raise RuntimeError("News index not found. Please run build_news_index() first.")

    # 1) Load index and metadata
    index = faiss.read_index(str(NEWS_INDEX_PATH))
    with open(NEWS_META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    # 2) Embed the query
    q_vec = embedder.encode([query], convert_to_numpy=True)

    # 3) Search
    D, I = index.search(q_vec, top_k)
    results: List[str] = []

    # 4) For each hit, build a compact snippet
    for idx in I[0]:
        entry = metadata[idx]
        title = entry.get("title", "")
        url = entry.get("url", "")
        pub = entry.get("publishedAt", "")
        snippet = f"{title} ({pub})\n{url}"
        results.append(snippet)

    return results


if __name__ == "__main__":
    # If run directly, rebuild from scratch
    build_news_index(reset=True)
