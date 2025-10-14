# retriever.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from langsmith.run_helpers import traceable, trace

COLL = "pdf_chunks"
EMB_NAME = "BAAI/bge-base-en-v1.5"

qc = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer(EMB_NAME)

# ---- BM25 snapshot (same as before)
def _snapshot(limit=20000):
    res = qc.scroll(COLL, with_payload=True, with_vectors=False, limit=limit)[0]
    texts = [r.payload["text"] for r in res]
    metas = [r.payload for r in res]
    tok = [t.split() for t in texts]
    return BM25Okapi(tok), texts, metas

_bm25, _texts, _metas = _snapshot()

def _dense_search(query, top_k):
    qv = embedder.encode(query).tolist()
    out = qc.search(COLL, query_vector=("dense", qv), limit=top_k, with_payload=True)
    return [{"text": r.payload["text"], "meta": r.payload, "score": float(r.score)} for r in out]

def _bm25_search(query, top_k):
    scores = _bm25.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"text": _texts[i], "meta": _metas[i], "bm25": float(scores[i])} for i in idx]

def _rrf_fuse(A, B, k):
    pool = {}
    for i, r in enumerate(A):
        key = (r["text"], r["meta"].get("doc_id"), r["meta"].get("page"))
        pool.setdefault(key, {"text": r["text"], "meta": r["meta"], "rrf": 0.0})
        pool[key]["rrf"] += 1.0 / (60 + i)
    for j, r in enumerate(B):
        key = (r["text"], r["meta"].get("doc_id"), r["meta"].get("page"))
        pool.setdefault(key, {"text": r["text"], "meta": r["meta"], "rrf": 0.0})
        pool[key]["rrf"] += 1.0 / (60 + j)
    fused = sorted(pool.values(), key=lambda x: x["rrf"], reverse=True)[:k]
    return fused

@traceable(run_type="retriever")          # <- TOP span: "retriever"
def hybrid_fuse_traced(query, top_k=12):
    # dense
    with trace("dense_search", run_type="tool", inputs={"query": query, "k": top_k*2}):
        A = _dense_search(query, top_k=top_k*2)
    # bm25
    with trace("bm25_search", run_type="tool", inputs={"query": query, "k": top_k*2}):
        B = _bm25_search(query, top_k=top_k*2)
    # fusion
    with trace("fusion", run_type="chain", inputs={"dense": len(A), "bm25": len(B)}):
        fused = _rrf_fuse(A, B, top_k)
    return fused
