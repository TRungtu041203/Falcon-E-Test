# query_expand.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langsmith.run_helpers import traceable, trace
import re, json, torch

MODEL = "tiiuae/falcon-E-1b-base"
device = 0 if torch.cuda.is_available() else -1

_tok = AutoTokenizer.from_pretrained(MODEL)
_mod = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if device == 0 else None
)
_gen = pipeline("text-generation", model=_mod, tokenizer=_tok, device=device)

def _generate(prompt, max_new_tokens=256):
    out = _gen(
        prompt, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=0.0, top_p=1.0, return_full_text=False,
        eos_token_id=_tok.eos_token_id
    )[0]["generated_text"].strip()
    return out

@traceable(run_type="chain")
def multi_query(query: str, n: int = 3):
    """Return up to n paraphrases/sub-questions."""
    tpl = (
        "Rewrite the user question into {n} diverse paraphrases or sub-questions "
        "that might retrieve different evidence. Output JSON array of strings.\n\n"
        f"User question: {query}\n\nJSON:"
    )
    with trace("multi_query_gen", run_type="llm", inputs={"query": query, "n": n}):
        raw = _generate(tpl.format(n=n))
    # Robust parse
    try:
        arr = json.loads(raw)
        cand = [s.strip() for s in arr if isinstance(s, str)]
    except Exception:
        # fallback: split lines / bullets
        cand = [s.strip("- •\t ") for s in raw.splitlines() if len(s.strip()) > 5]
    # Dedup near-identicals
    uniq, seen = [], set()
    for s in cand:
        k = re.sub(r"[^a-z0-9]+", "", s.lower())
        if k not in seen:
            uniq.append(s); seen.add(k)
    return uniq[:n]

@traceable(run_type="chain")
def step_back(query: str):
    """Return a higher-level reformulation that abstracts the task."""
    prompt = (
        "Step back: rewrite the question at a higher level to capture its core intent.\n"
        "Keep it concise and general.\n\n"
        f"Original: {query}\nStep-back:"
    )
    with trace("step_back_gen", run_type="llm", inputs={"query": query}):
        return _generate(prompt, max_new_tokens=128)

@traceable(run_type="chain")
def hyde_answer(query: str):
    """Generate a short hypothetical answer (HyDE)."""
    prompt = (
        "Hypothetical answer (grounded, factual tone, 4-8 sentences) to the question. "
        "Do NOT cite; just a plausible answer based on general knowledge.\n\n"
        f"Question: {query}\nHypothetical answer:"
    )
    with trace("hyde_gen", run_type="llm", inputs={"query": query}):
        return _generate(prompt, max_new_tokens=256)
