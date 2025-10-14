# gen_falcon.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, time
from langsmith.run_helpers import traceable, trace

MODEL = "tiiuae/falcon-E-1b-base"
device = 0 if torch.cuda.is_available() else -1

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if device == 0 else None
)
pipe = pipeline("text-generation", model=model, tokenizer=tok, device=device)

SYSTEM = (
    "You are a precise assistant for PDF QA. "
    "Answer ONLY from the provided context. If unknown, say you don't know. "
    "Cite sources as [doc_id:page]."
)

def _build_prompt(question: str, passages):
    context_block = "\n\n".join(
        f"[doc_id={p['meta'].get('doc_id','?')} page={p['meta'].get('page','?')}]\n{p['text']}"
        for p in passages
    )
    return (
        f"### System:\n{SYSTEM}\n\n"
        f"### Context:\n{context_block}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:"
    )

@traceable(run_type="llm")   # <- span: llm
def generate_answer_traced(query, passages):
    prompt = _build_prompt(query, passages)
    prompt_tokens = len(tok(prompt)["input_ids"])
    with trace(
        "falcon_generate",
        run_type="llm",
        inputs={"prompt_preview": prompt[:400]},
        metadata={"prompt_tokens": prompt_tokens, "passages": len(passages)}
    ):
        t0 = time.perf_counter()
        out = pipe(
            prompt,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            return_full_text=False,
            eos_token_id=tok.eos_token_id,
        )[0]["generated_text"].strip()
        dt = (time.perf_counter() - t0) * 1000
        # You can emit another nested span if you want to record latency separately:
        with trace("timing", run_type="tool", metadata={"latency_ms": round(dt, 1)}):
            pass
    return out
