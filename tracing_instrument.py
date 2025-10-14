# tracing_instrument.py (new small helper)
import os
from langsmith.run_helpers import traceable, trace
from langsmith import Client

client = Client()

@traceable(run_type="retriever")
def traced_retrieve(query, k, raw_retrieve_fn):
    return raw_retrieve_fn(query, top_k=k)

@traceable(run_type="chain")
def traced_rerank(query, passages, rerank_fn):
    return rerank_fn(query, passages)

@traceable(run_type="chain")
def traced_compress(passages, compress_fn):
    return compress_fn(passages)

@traceable(run_type="llm")
def traced_generate(query, passages, generate_fn):
    meta = {"prompt_passages": len(passages), "approx_chars": sum(len(p["text"]) for p in passages)}
    with trace("falcon_infer", run_type="llm", inputs={"query": query}, metadata=meta):
        return generate_fn(query, passages)
