import os, ast, threading
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
import numpy as np
from datasets import load_dataset
from huggingface_hub import InferenceClient

# -------------------------------
# Config
# -------------------------------
EMBED_COL = os.getenv("EMBED_COL", "embeddings_bge-m3")
DATASETS = [
    ("edouardfoussier/travail-emploi-clean", "train"),
    ("edouardfoussier/service-public-filtered", "train"),
]

HF_API_TOKEN   = os.getenv("HF_API_TOKEN")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
HF_LLM_MODEL   = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set. Add it in Space → Settings → Variables.")

# Try FAISS; fallback to NumPy if not available
_USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    _USE_FAISS = False

# -------------------------------
# Globals
# -------------------------------
_embed_client: Optional[InferenceClient] = None
_gen_client:   Optional[InferenceClient] = None

_index = None        # FAISS index or dense matrix (NumPy)
_payloads = None     # list[dict]
_dim = None
_lock = threading.Lock()

def _get_embed_client() -> InferenceClient:
    global _embed_client
    if _embed_client is None:
        _embed_client = InferenceClient(token=HF_API_TOKEN)
    return _embed_client

def _get_gen_client() -> InferenceClient:
    global _gen_client
    if _gen_client is None:
        _gen_client = InferenceClient(token=HF_API_TOKEN)
    return _gen_client

def _to_vec(x):
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    raise TypeError(f"Unsupported embedding type: {type(x)}")

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _embed_query(text: str) -> np.ndarray:
    # HF feature-extraction
    vec = _get_embed_client().feature_extraction(text, model=HF_EMBED_MODEL)
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 2:
        v = v[0]
    return _normalize(v)

def _load_datasets() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vecs, payloads = [], []
    for name, split in DATASETS:
        ds = load_dataset(name, split=split)
        for row in ds:
            v = _normalize(_to_vec(row[EMBED_COL]))
            vecs.append(v)
            p = dict(row); p.pop(EMBED_COL, None)
            payloads.append(p)
    X = np.stack(vecs, axis=0)
    return X, payloads

def _build_index() -> Tuple[Any, List[Dict[str, Any]], int]:
    X, payloads = _load_datasets()
    dim = X.shape[1]
    if _USE_FAISS:
        idx = faiss.IndexFlatIP(dim)
        idx.add(X)
    else:
        idx = X  # NumPy matrix
    return idx, payloads, dim

def _ensure_index():
    global _index, _payloads, _dim
    if _index is not None:
        return
    with _lock:
        if _index is None:
            _index, _payloads, _dim = _build_index()

def _search_numpy(X: np.ndarray, q: np.ndarray, k: int):
    scores = X @ q  # cosine/IP (normalized)
    k = min(k, len(scores))
    part = np.argpartition(-scores, k-1)[:k]
    order = part[np.argsort(-scores[part])]
    return scores[order], order

def retrieve(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    _ensure_index()
    q = _embed_query(query)
    if _USE_FAISS:
        D, I = _index.search(q[None, :], top_k)
        scores, idxs = D[0], I[0]
    else:
        scores, idxs = _search_numpy(_index, q, top_k)
    out = []
    for idx, sc in zip(idxs, scores):
        if idx == -1:
            continue
        p = _payloads[int(idx)]
        out.append({"score": float(sc), "payload": p})
    return out

def build_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    chunks = []
    for i, h in enumerate(passages, 1):
        p = h["payload"]
        text = p.get("text") or p.get("chunk_text") or ""
        source = p.get("source") or "unknown"
        title = p.get("title") or ""
        url = p.get("url") or ""
        chunks.append(f"[{i}] ({source}) {title}\n{text}\nURL: {url}\n")
    context = "\n\n".join(chunks)
    return f"""You are a helpful HR assistant. Answer the question strictly using the CONTEXT.
If the CONTEXT is not enough, say you don't know.

QUESTION:
{query}

CONTEXT:
{context}

Answer in French. Cite sources inline like [1], [2] where relevant.
"""

def stream_llm(prompt: str):
    # Stream tokens from HF Inference API text generation
    client = _get_gen_client()
    # temperature/params small so result is stable
    stream = client.text_generation(
        model=HF_LLM_MODEL,
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        stream=True,
        stop=None,
    )
    for chunk in stream:
        # chunk is a string token or piece; just yield it
        yield chunk

def format_sources(passages: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(passages, 1):
        p = h["payload"]
        title = (p.get("title") or "").strip() or "(Sans titre)"
        url = p.get("url") or ""
        src = p.get("source") or "unknown"
        lines.append(f"[{i}] **{title}** — _{src}_  " + (f"[lien]({url})" if url else ""))
    return "\n".join(lines)

# -------------------------------
# Gradio Chat handler
# -------------------------------
def respond(message, history):
    # Retrieve
    passages = retrieve(message, top_k=6)
    prompt = build_prompt(message, passages)

    # Stream answer
    answer_so_far = ""
    for token in stream_llm(prompt):
        answer_so_far += token
        yield answer_so_far

    # Append sources as an expandable block (return another message)
    sources_md = format_sources(passages)
    yield answer_so_far + "\n\n---\n**Sources**\n" + sources_md

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## 🔎 Assistant RH — RAG Chatbot")
    gr.Markdown(
        f"**Embeddings:** `{HF_EMBED_MODEL}` &nbsp;&nbsp;|&nbsp;&nbsp; **LLM:** `{HF_LLM_MODEL}`"
    )
    chat = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="Assistant RH",
        examples=[
            "Quels sont les droits à congés pour un agent contractuel ?",
            "Comment déclarer l’embauche d’un salarié (DPAE) ?",
            "Quelles sont les obligations de l’employeur pour le télétravail ?",
        ],
        retry_btn="Reformuler",
        undo_btn=None,
        clear_btn="Effacer",
        description="Posez une question RH. Réponse générée avec récupération documentaire.",
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=2).launch(server_name="0.0.0.0", server_port=7860)