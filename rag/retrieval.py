# retrieval.py
import os, ast, threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datasets import load_dataset
from huggingface_hub import InferenceClient


EMBED_COL = os.getenv("EMBED_COL", "embeddings_bge-m3")
DATASETS = [
    ("edouardfoussier/travail-emploi-clean", "train"),
    ("edouardfoussier/service-public-filtered", "train"),
]
HF_EMBED_MODEL = os.getenv("HF_EMBEDDINGS_MODEL", "BAAI/bge-m3")
HF_API_TOKEN  = os.getenv("HF_API_TOKEN")

_USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    _USE_FAISS = False

_embed_client: Optional[InferenceClient] = None
_index = None
_payloads = None
_lock = threading.Lock()

def _client() -> InferenceClient:
    global _embed_client
    if _embed_client is None:
        if not HF_API_TOKEN:
            raise RuntimeError("HF_API_TOKEN missing (.env)")
        _embed_client = InferenceClient(model=HF_EMBED_MODEL, token=HF_API_TOKEN)
    return _embed_client

def _to_vec(x):
    if isinstance(x, list): return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):  return np.asarray(ast.literal_eval(x), dtype=np.float32)
    raise TypeError(f"Bad embedding type: {type(x)}")

def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def embed(text: str) -> np.ndarray:
    vec = _client().feature_extraction(text)
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 2: v = v[0]
    return _norm(v)

def _load_corpus() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vecs, payloads = [], []
    for name, split in DATASETS:
        ds = load_dataset(name, split=split)
        for row in ds:
            v = _norm(_to_vec(row[EMBED_COL]))
            vecs.append(v)
            p = dict(row); p.pop(EMBED_COL, None)
            payloads.append(p)
    X = np.stack(vecs, axis=0)
    return X, payloads

CACHE = "/tmp/rag_index.npz"

def _faiss_from_X(X):
    import faiss
    idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X); return idx

def _build_index():
    # try cached index
    if os.path.exists(CACHE):
        try:
            d = np.load(CACHE, allow_pickle=True)
            X = d["X"]
            payloads = d["payloads"].tolist()
            return (_faiss_from_X(X) if _USE_FAISS else X), payloads
        except Exception:
            # cache corrupted → rebuild
            try: os.remove(CACHE)
            except: pass
    # build fresh
    X, payloads = _load_corpus()
    np.savez_compressed(CACHE, X=X, payloads=np.array(payloads, dtype=object))
    return (_faiss_from_X(X) if _USE_FAISS else X), payloads

def _ensure():
    global _index, _payloads
    if _index is not None: return
    with _lock:
        if _index is None:
            _index, _payloads = _build_index()

def _search_numpy(X: np.ndarray, q: np.ndarray, k: int):
    scores = X @ q
    k = max(1, min(k, len(scores)))
    part = np.argpartition(-scores, k-1)[:k]
    order = part[np.argsort(-scores[part])]
    return scores[order], order

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    _ensure()
    q = embed(query)
    if _USE_FAISS:
        D, I = _index.search(q[None, :], top_k)
        scores, idxs = D[0], I[0]
    else:
        scores, idxs = _search_numpy(_index, q, top_k)
    hits = []
    for i, s in zip(idxs, scores):
        if i == -1: continue
        p = _payloads[int(i)]
        hits.append({"score": float(s), "payload": p})
    return hits

# ---------- explicit warm-up helpers ----------
def warm_up_sync():
    try:
        _ = search("warmup", top_k=3)
    except Exception:
        pass

def warm_up_async():
    t = threading.Thread(target=warm_up_sync, daemon=True)
    t.start()

def ensure_ready():
    """Build the index once and warm the embedding endpoint."""
    _ensure()          # builds FAISS/NumPy index + loads payloads
    _ = embed("warmup")  # hits HF Inference API once to avoid cold-start
    
