import os
import json
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
import openai

# Paths & globals
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE = Path(__file__).parent / "faiss_index"
DOCS_DIR    = Path(__file__).parent.parent / "ingestion" / "data"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")

# Initialize models
embedder = SentenceTransformer(EMBED_MODEL)
openai.api_key = OPENAI_KEY

def build_faiss_index(reset: bool = False):
    """Read all classified filings, embed their chunks, and (re)build the FAISS index."""
    VECTOR_STORE.parent.mkdir(exist_ok=True)
    if reset and VECTOR_STORE.exists():
        VECTOR_STORE.unlink()

    # Collect text chunks
    texts = []
    metadata = []
    for file in DOCS_DIR.glob("*.xbrl"):  # or .htm/.html if you prefer
        content = file.read_text(encoding="utf-8", errors="ignore")
        # simple split into 500-word chunks
        for i, chunk in enumerate(content.split("\n\n")):
            texts.append(chunk)
            metadata.append({"source": file.name, "chunk_id": i})

    # Embed
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(VECTOR_STORE))

    # Save metadata
    with open(VECTOR_STORE.with_suffix(".meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def retrieve_context(question: str, top_k: int = 5) -> List[str]:
    """Given a question, pull top_k chunks from the FAISS store."""
    index = faiss.read_index(str(VECTOR_STORE))
    with open(VECTOR_STORE.with_suffix(".meta.json")) as f:
        metadata = json.load(f)

    q_vec = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    return [metadata[idx]["source"] + f" [chunk {metadata[idx]['chunk_id']}]" 
            + "\n\n" + 
            DOCS_DIR.joinpath(metadata[idx]["source"]).read_text()[metadata[idx]["chunk_id"]*500:(metadata[idx]["chunk_id"]+1)*500]
            for idx in I[0]]

def summarize_text(text: str) -> str:
    """Call GPT to generate a multi-level summary for a given text."""
    prompt = f"""
        You are an equity research assistant. Please generate:
        1) A one-sentence executive summary.
        2) Three bullet-point key highlights.
        Text:
        \"\"\"{text}\"\"\"
        """
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":"You summarize financial documents."},
                  {"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()

def answer_question(question: str) -> str:
    """Full RAG: retrieve context, then ask GPT to answer."""
    ctx = retrieve_context(question, top_k=5)
    joined = "\n\n---\n\n".join(ctx)
    prompt = f"""
        You are a financial analyst assistant. Use the following document excerpts to answer the question.
        Excerpts:
        {joined}

        Question: {question}
        """
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":"You answer financial queries based on provided context."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()
