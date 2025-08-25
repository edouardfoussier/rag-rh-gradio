import re
from collections import OrderedDict

CITATION_RE = re.compile(r"\[(\d+)\]")


def is_unknown_answer(txt: str) -> bool:
    """Detect 'no answer' / 'reformulate' replies."""
    s = (txt or "").lower()
    patterns = [
        "Je suis navré, je n'ai pas trouvé la réponse",
        "Je ne sais pas",
        "Je ne comprends pas"
        "Pourriez-vous reformuler",
        "je n'ai pas trouvé d'information pertinente",
    ]
    return any(p in s for p in patterns)


def _extract_cited_indices(text: str, k: int) -> list[int]:
    """Renvoie les indices (1..k) réellement cités dans le texte, sans doublon, ordonnés."""
    seen = OrderedDict()
    for m in CITATION_RE.finditer(text or ""):
        try:
            n = int(m.group(1))
            if 1 <= n <= k and n not in seen:
                seen[n] = True
        except Exception:
            pass
    return list(seen.keys())

def linkify_text_with_sources(text: str, passages: list[dict]) -> str:
    """
    Convertit [1], [2]… en vrais liens Markdown vers les sources.
    """
    import re
    mapping = {}
    for i, h in enumerate(passages, start=1):
        p = h.get("payload", h) or {}
        title = p.get("title") or p.get("url") or f"Source {i}"
        url = p.get("url")
        mapping[i] = (title, url)

    def _sub(m):
        idx = int(m.group(1))
        title, url = mapping.get(idx, (None, None))
        if url:
            # simple lien markdown [1](url)
            return f"[_[{idx}]_]({url} \"{title}\")"
        return m.group(0)

    return re.sub(r"\[(\d+)\]", _sub, text)

def _group_sources_md(passages: list[dict], used_idxs: list[int]) -> str:
    """
    Construit le markdown groupé :
    ### 📚 Sources (N)
    1. [Titre](url) _(extrait #1, 3)_
    2. [Autre](url2) _(extrait #2)_
    """
    if not passages:
        return "### 📚 Sources (0)\n_(aucune)_"

    # Utiliser les indices cités si dispo, sinon tomber sur 1..len(passages)
    if not used_idxs:
        used_idxs = list(range(1, len(passages) + 1))

    # Groupe par (url ou titre normalisé)
    groups = []  # [(key, title, url, [idxs])]
    key_to_pos = {}

    for idx in used_idxs:
        p = passages[idx-1]
        pl = p.get("payload", p) or {}
        title = (pl.get("title") or pl.get("url") or f"Source {idx}").strip()
        url = pl.get("url")

        key = (url or "").strip().lower() or title.lower()
        if key not in key_to_pos:
            key_to_pos[key] = len(groups)
            groups.append([key, title, url, []])
        groups[key_to_pos[key]][3].append(idx)

    # Ordonner chaque liste d'indices et construire le markdown
    lines = [f"### 📚 Sources ({len(groups)})"] if len(groups) > 1 else [f"### 📚 Source"]
    for i, (_, title, url, idxs) in enumerate(groups, start=1):
        idxs = sorted(idxs)
        idx_txt = ", ".join(map(str, idxs))
        label = "extrait" if len(idxs) == 1 else "extraits"
        suffix = f" _({label} # {idx_txt})_"
        if url:
            lines.append(f"{i}. [{title}]({url}){suffix}")
        else:
            lines.append(f"{i}. {title}{suffix}")
    return "\n".join(lines)

# Sanity Check
def stats():
    """Return quick information about the index and payloads."""
    _ensure()
    if _USE_FAISS:
        n = _index.ntotal
        dim = _index.d
    else:
        n = _index.shape[0]
        dim = _index.shape[1]
    return {
        "backend": "faiss" if _USE_FAISS else "numpy",
        "vectors": n,
        "dim": dim,
        "payloads": len(_payloads) if _payloads is not None else 0,
        "datasets": [f"{name}:{split}" for name, split in DATASETS],
    }

# def sources_markdown(passages: list[dict]) -> str:
#     if not passages:
#         return "### Sources\n_(aucune)_"

#     lines = [f"### 📚 Sources ({len(passages)})"]
#     for i, h in enumerate(passages, start=1):
#         p = h.get("payload", h) or {}
#         title = (p.get("title") or p.get("url") or f"Source {i}").strip()
#         url = p.get("url")
#         score = h.get("score")
#         # snippet = (p.get("text") or "").strip().replace("\n", " ")

#         # # on coupe le snippet pour pas que ce soit trop long
#         # if len(snippet) > 180:
#         #     snippet = snippet[:180] + "…"

#         # ligne principale
#         if url:
#             line = f"{i}. [{title}]({url})"
#         else:
#             line = f"{i}. {title}"

#         # on ajoute le score et snippet en italique, plus discrets
#         if isinstance(score, (int, float)):
#             line += f" _(score {score:.3f})_"
#         # if snippet:
#         #     line += f"\n   > {snippet}"

#         lines.append(line)

#     return "\n".join(lines)
