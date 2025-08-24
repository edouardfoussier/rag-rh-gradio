import os
import ast
import json
import threading
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
import numpy as np
from datasets import load_dataset
from huggingface_hub import InferenceClient

# ------------------
# Config
# ------------------
EMBED_COL = os.getenv("EMBED_COL", "embeddings_bge-m3")
DATASETS = [
    ("edouardfoussier/travail-emploi-clean", "train"),
    ("edouardfoussier/service-public-filtered", "train"),
]

HF_EMBED_MODEL = os.getenv("HF_EMBEDDINGS_MODEL", "BAAI/bge-m3")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # set in Space → Settings → Variables

# Optional: limit rows per dataset to keep RAM in check while testing
MAX_ROWS = int(os.getenv("MAX_ROWS_PER_DATASET", "0"))  # 0 = no limit

# Try FAISS; fall back to NumPy
_USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    _USE_FAISS = False

# ------------------
# Embedding client
# ------------------
_embed_client: Optional[InferenceClient] = None
def _get_embed_client() -> InferenceClient:
    global _embed_client
    if _embed_client is None:
        mid = HF_EMBED_MODEL.strip()

        # Auto-fix very common bad value like "sentence-transformers/BAAI/bge-m3"
        if mid.lower().startswith("sentence-transformers/baai/"):
            mid = mid.split("/", 1)[1]  # -> "BAAI/bge-m3"

        if mid.count("/") != 1:
            raise ValueError(
                f"HF_EMBEDDINGS_MODEL must be 'owner/name', got '{mid}'. "
                "Examples: 'BAAI/bge-m3', 'sentence-transformers/all-MiniLM-L6-v2'."
            )
        if not HF_API_TOKEN:
            raise RuntimeError(
                "HF_API_TOKEN is not set. Go to Space → Settings → Variables and add HF_API_TOKEN (a WRITE token)."
            )
        _embed_client = InferenceClient(model=mid, token=HF_API_TOKEN, repo_type="model")
    return _embed_client

# ------------------
# Vector helpers
# ------------------
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
    vec = _get_embed_client().feature_extraction(text)
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 2:
        v = v[0]
    return _normalize(v)

# ------------------
# Index storage
# ------------------
_index = None           # faiss index or raw matrix (np.ndarray)
_payloads: List[Dict[str, Any]] = []
_dim = None
_lock = threading.Lock()

def _load_datasets() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vecs, payloads = [], []
    for name, split in DATASETS:
        ds = load_dataset(name, split=split)
        if MAX_ROWS > 0:
            ds = ds.select(range(min(MAX_ROWS, len(ds))))
        for row in ds:
            v = _normalize(_to_vec(row[EMBED_COL]))
            vecs.append(v)
            p = dict(row)
            p.pop(EMBED_COL, None)
            payloads.append(p)
    X = np.stack(vecs, axis=0) if vecs else np.zeros((0, 1), dtype=np.float32)
    return X, payloads

def _build_index() -> Tuple[Any, List[Dict[str, Any]], int]:
    X, payloads = _load_datasets()
    if X.size == 0:
        return (np.zeros((0, 1), dtype=np.float32), payloads, 1)
    dim = X.shape[1]
    if _USE_FAISS:
        idx = faiss.IndexFlatIP(dim)
        idx.add(X)
    else:
        idx = X  # NumPy fallback
    return idx, payloads, dim

def _ensure_index_loaded():
    global _index, _payloads, _dim
    if _index is not None:
        return
    with _lock:
        if _index is not None:
            return
        idx, pls, d = _build_index()
        _index, _payloads, _dim = idx, pls, d

def _search_ip_numpy(X: np.ndarray, q: np.ndarray, k: int):
    # Both normalized => inner product = cosine similarity
    scores = X @ q
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    return scores[order], order

def _search(query: str, k: int, source_filter: Optional[str]) -> List[Dict[str, Any]]:
    _ensure_index_loaded()
    if _dim is None or (_USE_FAISS and _index.ntotal == 0) or (not _USE_FAISS and _index.shape[0] == 0):
        return []
    q = _embed_query(query)
    if _USE_FAISS:
        D, I = _index.search(q[None, :], k)
        scores, idxs = D[0], I[0]
    else:
        scores, idxs = _search_ip_numpy(_index, q, k)
    out = []
    for idx, sc in zip(idxs, scores):
        if int(idx) < 0:
            continue
        pl = _payloads[int(idx)]
        if source_filter and pl.get("source") != source_filter:
            continue
        out.append({
            "id": str(int(idx)),
            "score": float(sc),
            "title": (pl.get("title") or "").strip() or "(Sans titre)",
            "url": pl.get("url") or "",
            "source": pl.get("source") or "",
            "snippet": (pl.get("text") or pl.get("chunk_text") or "")[:500]
        })
    return out

# ------------------
# Gradio UI
# ------------------
def do_search(query, source, top_k):
    try:
        if not query or not query.strip():
            return gr.update(value="<i>Entrez une question…</i>", visible=True)
        src_filter = None if (not source or source == "(Tous)") else source
        hits = _search(query.strip(), int(top_k), src_filter)
        if not hits:
            return gr.update(value="<b>0 résultat</b>", visible=True)

        lines = [f"<b>Top {len(hits)} résultats</b><br>"]
        for i, h in enumerate(hits, 1):
            badge = {
                "travail-emploi": '<span style="background:#2563eb;color:white;padding:2px 6px;border-radius:999px;font-size:12px">travail-emploi</span>',
                "service-public": '<span style="background:#059669;color:white;padding:2px 6px;border-radius:999px;font-size:12px">service-public</span>',
            }.get(h["source"].lower(), f'<span style="background:#6b7280;color:white;padding:2px 6px;border-radius:999px;font-size:12px">{h["source"] or "unknown"}</span>')
            title = h["title"]
            url = h["url"]
            score = f"{h['score']:.3f}"
            head = f"#{i} {badge} "
            head += f'<a href="{url}" target="_blank">{title}</a>' if url else title
            lines.append(f"{head} &nbsp;&nbsp; <code>cos={score}</code><br>")
            if h["snippet"]:
                lines.append(f"<div style='margin-left:1rem;color:#444'>{h['snippet']}</div><br>")
        return gr.update(value="\n".join(lines), visible=True)
    except Exception as e:
        return gr.update(value=f"<b>Erreur:</b> {e}", visible=True)

with gr.Blocks(title="RAG-RH (Gradio)") as demo:
    gr.Markdown("## 🔎 Assistant RH — RAG Demo (Gradio)")
    with gr.Row():
        query = gr.Textbox(label="Votre question", placeholder="Posez votre question…", scale=3)
        run = gr.Button("Rechercher", variant="primary", scale=1)
    with gr.Row():
        source = gr.Dropdown(choices=["(Tous)", "travail-emploi", "service-public"],
                             value="(Tous)", label="Filtre", scale=1)
        topk = gr.Slider(3, 30, value=8, step=1, label="Top-K", scale=2)

    out = gr.HTML(visible=False)

    run.click(do_search, inputs=[query, source, topk], outputs=out)
    query.submit(do_search, inputs=[query, source, topk], outputs=out)

if __name__ == "__main__":
    # For local testing: `python app.py`
    demo.launch(server_name="0.0.0.0", server_port=7860)