# rag/synth.py
import os
from openai import OpenAI
from rag.utils import utf8_safe
from datetime import date


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

def _build_prompt(query, passages):

    # Construire des blocs numérotés et balisés
    blocks = []
    for i, h in enumerate(passages, start=1):
        p = h.get("payload", h) or {}
        title = (p.get("title") or p.get("url") or f"Source {i}").strip()
        url   = p.get("url") or ""
        text  = utf8_safe(p.get("text") or "")
        # Chaque bloc porte explicitement son index [i]
        blocks.append(
            f"### Source [{i}] — {title}\n"
            f"{('URL: ' + url) if url else ''}\n"
            f"{text}\n"
        )

    context = "\n\n".join(blocks)
    query = utf8_safe(query)
    today = date.today().strftime("%d %B %Y")  # e.g. "27 août 2025"


    return (
        "Tu es un assistant RH chargé de répondre à des questions dans le domaine des ressources humaines en t'appuyant sur les sources fournies.\n"
        "La date d'aujourd'hui est : {today}.\n\n"
        "Consignes :\n"
        "- Réponds de manière factuelle, concise et polie (vouvoiement).\n"
        "- Quand tu affirmes un fait, cite tes sources en fin de phrase avec le format [1], [2]… en te basant sur l'index de ces sources (ex: [1] est la source 1, [2] est la source 2, etc.)\n\n"
        "- Si l'information n'est pas présente dans les sources, réponds : \"Je suis navré, je n'ai pas trouvé la réponse à cette question\".\n\n"
        "- Si la question est mal formulée, réponds : \"Je ne comprends pas la question. Pourriez-vous reformuler ?\"\n\n"
        "- Ne fabrique pas de liens ni de références.\n\n"
        f"Question: {query}\n"
        f"Sources (indexées) : {context}\n\n"
        "Réponse:"
    )

def synth_answer_stream(query, passages):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=LLM_BASE_URL)
    prompt = utf8_safe(_build_prompt(query, passages))

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
    )

    for event in stream:
        if not getattr(event, "choices", None):
            continue
        delta = event.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield utf8_safe(delta.content or "")