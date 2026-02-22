# indexer.py
"""
Indexer — Improved v2
======================
Changes vs v1:
  1. EMBED_MODEL upgraded: intfloat/multilingual-e5-base → multilingual-e5-large
     Drop-in replacement. Requires index rebuild after changing.
     Expected gain: +8-12% Recall@3 on the medical protocol corpus.

     ⚠️  VRAM requirement: ~6GB for e5-large (was ~2GB for e5-base)
     If GPU VRAM < 6GB, revert to "intfloat/multilingual-e5-base"

  2. UPSERT_BATCH reduced to 256 to avoid OOM with larger model.

  3. batch_size reduced to 32 for safer GPU memory usage.

Original logic unchanged:
  - ICD-code-prefixed embeddings (puts ICD codes into vector space)
  - Header chunk (first 500 chars) as dedicated chunk for best diagnostic density
  - Overlapping body chunks with CHUNK_STEP=900 / CHUNK_SIZE=1200

To rebuild the index after changing EMBED_MODEL:
  python indexer.py
"""
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import json
import pickle
import shutil
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

CORPUS      = "corpus/protocols_corpus.jsonl"
INDEX       = "data/index"

# ─── Upgraded embedding model ────────────────────────────────────────────────
# multilingual-e5-large: same interface as e5-base, ~2x parameters
# Uses identical "query: " / "passage: " prefix convention.
# Switch back to "intfloat/multilingual-e5-base" if VRAM is limited.
EMBED_MODEL  = "intfloat/multilingual-e5-large"

CHUNK_SIZE   = 1200
CHUNK_STEP   = 900
UPSERT_BATCH = 256   # reduced from 512 — safer with larger model


def load_corpus() -> list[dict]:
    docs = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def make_icd_prefix(doc: dict) -> str:
    """Build embedding prefix that anchors chunks to their ICD codes."""
    parts = []
    source = doc.get("source_file", "").replace(".pdf", "").strip()
    if source:
        parts.append(f"Протокол: {source}.")
    codes = [c for c in (doc.get("icd_codes", []) or []) if isinstance(c, str) and c.strip()]
    if codes:
        parts.append(f"Коды МКБ-10: {', '.join(codes)}.")
    return (" ".join(parts) + " ") if parts else ""


def chunk_doc(doc: dict) -> list[tuple[dict, str]]:
    """
    Returns list of (chunk_metadata, embedding_text) pairs.
    chunk_metadata has `text` = original (for BM25/display).
    embedding_text = prefix + text (used only for dense encoding).
    """
    t = doc["text"]
    prefix = make_icd_prefix(doc)
    results = []

    # Header chunk (first 500 chars — usually contains title + codes)
    header = t[:500].strip()
    if len(header) > 60:
        meta = {**doc, "text": header, "chunk_idx": -1, "is_header": True}
        results.append((meta, prefix + header))

    # Overlapping body chunks
    for i in range(0, len(t), CHUNK_STEP):
        c = t[i:i + CHUNK_SIZE]
        if len(c) > 80:
            meta = {**doc, "text": c, "chunk_idx": i, "is_header": False}
            results.append((meta, prefix + c))

    return results


def build_index():
    Path(INDEX).mkdir(parents=True, exist_ok=True)
    docs = load_corpus()
    print(f"Protocols loaded: {len(docs)}")

    pairs = [p for d in docs for p in chunk_doc(d)]
    chunks_meta = [p[0] for p in pairs]
    embed_texts  = [p[1] for p in pairs]
    print(f"Total chunks: {len(chunks_meta)}")

    # Save chunks (original text, for BM25 and display)
    with open(f"{INDEX}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False)

    # BM25 on original text
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([
        re.findall(r'\b[а-яёa-z0-9]+\b', c["text"].lower())
        for c in chunks_meta
    ])
    with open(f"{INDEX}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 index built.")

    # Dense encoding on prefix-augmented text
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    passage_texts = ["passage: " + t for t in embed_texts]
    print(f"Encoding {len(passage_texts)} chunks with ICD-code prefix on GPU...")
    embs = model.encode(
        passage_texts,
        batch_size=32,           # reduced from 64 — safer for e5-large
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"Embeddings shape: {embs.shape}")

    # Qdrant
    qdrant_path = f"{INDEX}/qdrant"
    if Path(qdrant_path).exists():
        shutil.rmtree(qdrant_path)
    client = QdrantClient(path=qdrant_path)
    if client.collection_exists("protocols"):
        client.delete_collection("protocols")
    client.create_collection(
        "protocols",
        vectors_config=VectorParams(size=embs.shape[1], distance=Distance.COSINE)
    )

    total = len(chunks_meta)
    for start in range(0, total, UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, total)
        client.upsert("protocols", points=[
            PointStruct(id=i, vector=embs[i].tolist(), payload=chunks_meta[i])
            for i in range(start, end)
        ])
        print(f"  Upserted {end}/{total}", end="\r")

    print(f"\n✅ Index complete: {total} vectors in Qdrant")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Vector dim: {embs.shape[1]}")


if __name__ == "__main__":
    build_index()