import os
import json
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
import openai
from sentiment.finbert import sentiment_score

# Paths & globals
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE = Path(__file__).parent / "faiss_index"
METADATA_STORE = VECTOR_STORE.with_suffix(".meta.json")
DOCS_DIR    = Path(__file__).parent.parent / "ingestion" / "data"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 1000

# Initialize models
embedder = SentenceTransformer(EMBED_MODEL)
openai.api_key = OPENAI_KEY


def build_faiss_index(reset: bool = False, chunk_size: int = 1000):
    """
    Read all filings, split into chunks, compute sentiment, embed, and build the FAISS index.
    """
    VECTOR_STORE.parent.mkdir(exist_ok=True)
    if reset and VECTOR_STORE.exists():
        VECTOR_STORE.unlink()
    texts = []
    metadata = []
    exts = ['.xbrl', '.html', '.htm']
    for file in DOCS_DIR.iterdir():
        if file.suffix.lower() in exts:
            content = file.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                texts.append(chunk)
                # Compute sentiment for each chunk
                sent = sentiment_score(chunk)
                metadata.append({
                    "source": file.name,
                    "chunk_index": i // chunk_size,
                    "sentiment": sent.get("label"),
                    "sentiment_score": sent.get("score")
                })
    if not texts:
        print("No documents found to index.")
        return
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(VECTOR_STORE))
    with open(METADATA_STORE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Built FAISS index with {len(texts)} chunks and sentiment data.")


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
