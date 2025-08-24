import os
from openai import OpenAI

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

def _build_prompt(query, passages):
    ctx = "\n\n".join([(p["payload"].get("text") or "") for p in passages])
    return (
        "Tu es un assistant RH de la fonction publique française.\n"
        "- Réponds de façon factuelle et concise.\n"
        "- Cite les sources en fin de phrase avec [1], [2]… basées sur l’ordre des passages.\n"
        "- Si l’info n’est pas dans les sources, réponds « Je ne sais pas ».\n\n"
        f"Question: {query}\n\nSources (indexées):\n{ctx}\n\nRéponse:"
    )

def synth_answer_stream(query, passages):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=LLM_BASE_URL)
    prompt = _build_prompt(query, passages)
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,  # 👈 IMPORTANT
    )
    # The SDK yields events with deltas
    for event in stream:
        delta = getattr(getattr(event, "choices", [None])[0], "delta", None)
        if delta and delta.content:
            yield delta.content

# def linkify(text, passages):
#     # (optional) keep simple: return text as-is for now
#     return text

def render_sources(passages):
    lines = []
    for i, p in enumerate(passages, 1):
        title = (p["payload"].get("title") or "").strip() or "Sans titre"
        url = p["payload"].get("url") or ""
        lines.append(f"[{i}] {title}{' – ' + url if url else ''}")
    return "\n".join(lines)

# def linkify_text_with_sources(text: str, passages):
#     """
#     Replace [1], [2]... with clickable links if the passage has a URL.
#     Also append a Sources section as a numbered list.
#     """
#     # Build a map: 1-based index -> url
#     urls = []
#     for p in passages:
#         url = (p["payload"].get("url") or "").strip()
#         urls.append(url if url.startswith("http") else "")

#     # Inline [n] -> [n](url) when available
#     out = text
#     for i, url in enumerate(urls, start=1):
#         if url:
#             out = out.replace(f"[{i}]", f"[{i}]({url})")

#     # Add a Sources section
#     lines = ["\n\n---\n**Sources**"]
#     for i, p in enumerate(passages, start=1):
#         title = (p["payload"].get("title") or "").strip() or "Sans titre"
#         url = (p["payload"].get("url") or "").strip()
#         if url.startswith("http"):
#             lines.append(f"{i}. [{title}]({url})")
#         else:
#             lines.append(f"{i}. {title}")
#     return out + "\n" + "\n".join(lines)
# import os
# from openai import OpenAI

# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

# def _first_k_chars(text, k=1200):
#     t = text.strip()
#     return t[:k] + ("…" if len(t) > k else "")

# def _build_prompt(query, passages):
#     chunks = []
#     for i, p in enumerate(passages, 1):
#         txt = p["payload"].get("text") or ""
#         chunks.append(f"[{i}] {_first_k_chars(txt)}")

# # def _build_prompt(query, passages):
# #     chunks = []
# #     for i, p in enumerate(passages, 1):
# #         txt = p["payload"].get("text") or ""
#         # chunks.append(f"[{i}] {txt}")
#     context = "\n\n".join(chunks)

#     return f"""Tu es un assistant RH de la fonction publique française.
# - Réponds de manière factuelle et concise.
# - Cite tes sources en fin de phrase avec [n] correspondant aux extraits ci-dessous.
# - Si l’information n’est pas dans les sources, réponds : “Je ne sais pas”.
# - Ne fabrique pas de liens ni de références.

# Question: {query}

# Extraits indexés:
# {context}

# Réponse:"""

# def synth_answer_stream(query, passages):
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=LLM_BASE_URL)
#     prompt = _build_prompt(query, passages)

#     # ✅ Correct streaming usage
#     stream = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#         stream=True,  # <- this is key
#     )
#     for chunk in stream:
#         delta = getattr(chunk.choices[0].delta, "content", None)
#         if delta:
#             acc.append(delta)
#             yield delta  # stream piece by piece
# # def synth_answer(query, passages):
# #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=LLM_BASE_URL)
# #     prompt = _build_prompt(query, passages)

# #     resp = client.chat.completions.create(
# #         model=LLM_MODEL,
# #         messages=[{"role": "user", "content": prompt}],
# #         temperature=0.2,
# #     )
# #     return resp.choices[0].message.content.strip()

# # --- HELPERS

# def render_sources(passages):
#     lines = []
#     for i, p in enumerate(passages, 1):
#         pl = p["payload"]
#         title = (pl.get("title") or "Source").strip()
#         url = pl.get("url") or ""
#         lines.append(f"[{i}] {title}" + (f" — {url}" if url else ""))
#     return "\n".join(lines)

# def linkify(text, passages):
#     # turn [1] -> markdown link when url exists
#     for i, p in enumerate(passages, 1):
#         url = p["payload"].get("url")
#         if url:
#             text = text.replace(f"[{i}]", f"[{i}]({url})")
#     return text






