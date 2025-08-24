import re

def linkify_text_with_sources(text: str, passages: list[dict]) -> str:
    """
    Convert [1], [2]… in `text` to markdown links using the corresponding
    passage payloads (expects top-5 `hits` from your retriever).
    """
    # Build mapping: 1-based index -> (title, url)
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
            # turn [n] into [n](url "title")
            return f"[{idx}]({url} \"{title}\")"
        # leave as plain [n] if no URL
        return m.group(0)

    return re.sub(r"\[(\d+)\]", _sub, text)