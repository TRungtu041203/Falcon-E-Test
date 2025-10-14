# rerank_compress.py
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from langsmith.run_helpers import traceable, trace
import re, numpy as np

_ce = CrossEncoder("BAAI/bge-reranker-base")

@traceable(run_type="chain")     # <- span: rerank
def rerank_traced(query, passages, top_k=6):
    with trace("cross_encoder", run_type="tool", inputs={"pairs": len(passages)}):
        pairs = [(query, p["text"]) for p in passages]
        scores = _ce.predict(pairs)  # numpy array
    ranked = sorted(zip(passages, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
    return [p for p,_ in ranked]

def _is_low_signal(line: str) -> bool:
    s = line.strip().lower()
    if len(s) < 8: return True
    if re.match(r'^(\d+(\.\d+)*\s+)+', s): return True
    if s.startswith("table of contents"): return True
    if sum(ch.isdigit() for ch in s) > 0.4*len(s): return True
    return False

@traceable(run_type="chain")     # <- span: compress
def compress_traced(passages, max_chars=2800):
    kept, used = [], 0
    with trace("extractive_compress", run_type="tool", inputs={"passages": len(passages), "budget": max_chars}):
        for p in passages:
            for sent in re.split(r'(?<=[.!?])\s+', p["text"]):
                if _is_low_signal(sent): 
                    continue
                if used + len(sent) + 1 > max_chars:
                    break
                kept.append(sent.strip())
                used += len(sent) + 1
            if used >= max_chars:
                break
    return "\n".join(kept)
