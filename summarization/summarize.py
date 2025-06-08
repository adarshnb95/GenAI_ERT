import os
import json
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
import openai
from sentiment.finbert import sentiment_score
from dotenv import load_dotenv

# Paths & globals
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE = Path(__file__).parent / "faiss_index"
METADATA_STORE = VECTOR_STORE.with_suffix(".meta.json")
DOCS_DIR    = Path(__file__).parent.parent / "ingestion" / "data"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 1000

# Initialize models
embedder = SentenceTransformer(EMBED_MODEL)
load_dotenv() 
openai.api_key = OPENAI_KEY

 # Load environment variables from .env file

_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def build_faiss_index_for_ticker(
    ticker: str,
    reset: bool = False,
    chunk_size: int = 1000
):
    ticker = ticker.upper()

    # 1) Source docs folder
    docs_dir = Path(__file__).parent.parent / "ingestion" / "data" / ticker
    if not docs_dir.exists():
        raise FileNotFoundError(f"No docs for {ticker}. Run fetch_for_ticker first.")

    # 2) Base path for all indexes
    base_index_dir = Path(__file__).parent / "faiss_index"
    # If someone accidentally committed a file named 'faiss_index', remove it
    if base_index_dir.exists() and not base_index_dir.is_dir():
        base_index_dir.unlink()

    # 3) Per-ticker output directory
    out_dir = base_index_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path  = out_dir / f"{ticker}.index"
    meta_path = out_dir / f"{ticker}.meta.json"
    if reset and idx_path.exists():
        idx_path.unlink()

    texts, metadata = [], []
    for file in docs_dir.iterdir():
        if file.suffix.lower() not in (".html", ".htm", ".xml", ".xbrl"):
            continue
        content = file.read_text(encoding="utf-8", errors="ignore").strip()
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            texts.append(chunk)
            metadata.append({
                "source": file.name,
                "chunk_index": i // chunk_size
            })

    if not texts:
        print(f"[summarize] No chunks for {ticker}.")
        return

    # 3) embed
    embedder = _get_embedder()
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # 4) build FAISS
    import faiss
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(idx_path))

    # 5) write metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[summarize] Built FAISS index for {ticker}: {len(texts)} chunks.")

def retrieve_context_for_ticker(
    ticker: str,
    question: str,
    top_k: int = 5,
    chunk_size: int = 1000
):
    import json, faiss
    from pathlib import Path

    ticker = ticker.upper()
    docs_dir = Path(__file__).parent.parent / "ingestion" / "data" / ticker
    vec_dir  = Path(__file__).parent / "faiss_index" / ticker
    idx_path = vec_dir / f"{ticker}.index"
    meta_path= vec_dir / f"{ticker}.meta.json"
    if not idx_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Index for {ticker} not found – run build_faiss_index_for_ticker().")

    # load
    index = faiss.read_index(str(idx_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # embed query
    q_vec = _get_embedder().encode([question], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)

    snippets = []
    for hit in I[0]:
        m = metadata[hit]
        src_file = docs_dir / m["source"]
        full = src_file.read_text(encoding="utf-8", errors="ignore")
        start = m["chunk_index"] * chunk_size
        snippet = full[start : start + chunk_size]
        snippets.append(f"Source: {m['source']} (chunk {m['chunk_index']})\n{snippet}")

    return snippets

def retrieve_context(question: str, top_k: int = 5) -> List[str]:
    """
    Given a question, retrieve top_k chunks from the FAISS store.
    """
    if not VECTOR_STORE.exists() or not METADATA_STORE.exists():
        raise RuntimeError("FAISS index not found. Please run build_faiss_index() first.")
    index = faiss.read_index(str(VECTOR_STORE))
    with open(METADATA_STORE, encoding="utf-8") as f:
        metadata = json.load(f)
    q_vec = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    contexts = []
    for idx in I[0]:
        entry = metadata[idx]
        src = entry["source"]
        chunk_idx = entry["chunk_index"]
        full = (DOCS_DIR / src).read_text(encoding="utf-8", errors="ignore")
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        snippet = full[start:end]
        contexts.append(
            f"Source: {src}, chunk {chunk_idx}, sentiment: {entry['sentiment']} ({entry['sentiment_score']:.2f})\n{snippet}"
        )
    return contexts


def summarize_text(text: str) -> str:
    """
    Generate a one-sentence summary and three bullet-point highlights via GPT.
    """
    prompt = f"""
        You are an equity research assistant. Please generate:
        1) A one-sentence executive summary.
        2) Three bullet-point key highlights.
        Text:
        {text}
        """
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You summarize financial documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()


def answer_question(question: str) -> str:
    """
    Full RAG: retrieve context chunks, then ask GPT to answer the question.
    """
    contexts = retrieve_context(question)
    joined = "\n\n---\n\n".join(contexts)
    prompt = f"""
        You are a financial analyst assistant. Use the following document excerpts to answer the question.
        Excerpts:
        {joined}

        Question: {question}
        """
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You answer financial queries based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()
