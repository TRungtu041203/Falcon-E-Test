# app.py
from langsmith.run_helpers import trace
from retriever import hybrid_fuse_traced
from rerank_compress import rerank_traced, compress_traced
from gen_falcon import generate_answer_traced

def answer_query(query: str):
    with trace("answer_query", run_type="chain", inputs={"query": query}):
        raw = hybrid_fuse_traced(query, top_k=12)                # retriever
        top = rerank_traced(query, raw, top_k=6)                 # rerank
        ctx = compress_traced(top, max_chars=2800)               # compress
        passages = [{"text": ctx, "meta": {"doc_id":"mixed","page":"various"}}] if ctx else top
        return generate_answer_traced(query, passages)           # llm

if __name__ == "__main__":
    q = "Describe the id-service and how it works in the LIACARA build."
    print(answer_query(q))
