import os
import gradio as gr
from gradio import update as gr_update  # tiny alias
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv(override=True)

from rag.retrieval import search, ensure_ready
from rag.synth import synth_answer_stream
from helpers import _extract_cited_indices, linkify_text_with_sources, _group_sources_md, is_unknown_answer, _last_user_and_assistant_idxs


# ---------- Warm-Up ----------

def _warmup():
    try:
        ensure_ready()
        return "✅ Modèles initialisés !"
    except Exception as e:
        return f"⚠️ Warmup a échoué : {e}"


# ---------- Chat step 1: add user message ----------
def add_user(user_msg: str, history: list[dict]):
    """
    history (messages mode) looks like:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
    We append the user's message, then an empty assistant message to stream into.
    """
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "", history

    new_history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},  # placeholder for streaming
    ]
    return "", new_history


# ---------- Chat step 2: stream assistant answer ----------
def bot(history: list[tuple], api_key: str, top_k: int, model_name: str, temperature: float):
    """
    Streaming generator for messages-format history.
    Yields (updated_history, sources_markdown).
    """
    # Initial sources panel content
    empty_sources = "### 📚 Sources\n_Ici, vous pourrez consulter les sources utilisées pour formuler la réponse._"

    if not history:
        yield history, empty_sources
        return

    # Inject BYO key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key.strip()

    # Identify the pair (user -> assistant placeholder)
    try:
        u_idx, a_idx = _last_user_and_assistant_idxs(history)
    except Exception:
        yield history, empty_sources
        return
    
    user_msg = history[u_idx]["content"]

    # Retrieval
    k = int(max(top_k, 1))
    try:
        hits = search(user_msg, top_k=k)
    except Exception as e:
        history[a_idx]["content"] = f"❌ Retrieval error: {e}"
        yield history, empty_sources
        return

    # Show a small “thinking” placeholder immediately
    history[a_idx]["content"] = "⏳ Synthèse en cours…"
    yield history, empty_sources

    # Streaming LLM
    acc = ""
    try:
        for chunk in synth_answer_stream(user_msg, hits[:k], model=model_name, temperature=temperature):
            acc += chunk or ""
            history[a_idx]["content"] = acc
            # Stream without sources first (or keep a lightweight panel if you prefer)
            yield history, empty_sources
    except Exception as e:
        history[a_idx]["content"] = f"❌ Synthèse: {e}"
        yield history, empty_sources
        return

    # Finalize + linkify citations
    acc_linked = linkify_text_with_sources(acc, hits[:k])
    history[a_idx]["content"] =  acc_linked
    
    # Decide whether to show sources
    if is_unknown_answer(acc_linked):
        # No sources for unknown / reformulate
        yield history, empty_sources
        return
    
    # Construit la section sources à partir des citations réelles [n]
    used = _extract_cited_indices(acc_linked, k)
    if not used:
        yield history, "### 📚 Sources\n_Ici, vous pourrez consulter les sources utilisées pour formuler la réponse._"
        return
    
    grouped_sources = _group_sources_md(hits[:k], used)
    yield history, gr_update(visible=True, value=grouped_sources)
    # yield history, sources_md


# ---------- UI ----------
with gr.Blocks(theme="soft", fill_height=True) as demo:
    gr.Markdown("# 🇫🇷 Assistant RH — RAG Chatbot")
            # Warmup status (put somewhere visible)
    status = gr.Markdown("⏳ Initialisation des modèles du RAG…")

    # Sidebar (no 'label' arg)
    with gr.Sidebar(open=True):
        gr.Markdown("## ⚙️ Paramètres")
        api_key = gr.Textbox(
            label="🔑 OpenAI API Key (BYOK — never stored)",
            type="password",
            placeholder="sk-… (optional if set in env)"
        )
        # let user choose the OpenAI model
        model = gr.Dropdown(
            label="⚙️ OpenAI model",
            choices=[
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-3.5-turbo",
            ],
            value="gpt-4o-mini"
        )
        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K passages")
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,  # valeur par défaut
            step=0.1,
            label="Température du modèle"
        )
    with gr.Row():
        with gr.Column(scale=4):
            chat = gr.Chatbot(
                label="Chat Interface",
                height="65vh",
                show_copy_button=False,
                avatar_images=(
                    "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/icons/huggingface-logo.svg",
                    "assets/chatbot.png",
                ),
                render_markdown=True,
                show_label=False,
                type="messages",
                placeholder="<p style='text-align: center;'>Bonjour 👋,</p><p style='text-align: center;'>Je suis votre assistant HR. Je me tiens prêt à répondre à vos questions.</p>"
            )
            # input row
            with gr.Row(equal_height=True):
                msg = gr.Textbox(
                    placeholder="Posez votre question…",
                    show_label=False,
                    scale=5,
                )
                send = gr.Button("Envoyer", variant="primary", scale=1)

        with gr.Column(scale=1):
            sources = gr.Markdown("### 📚 Sources\n_Ici, vous pourrez consulter les sources utilisées pour formuler la réponse._")

    state = gr.State([])  # chat history: list[tuple(user, assistant)]

    # wire events: user submits -> add_user -> bot streams
    send_click = send.click(add_user, [msg, state], [msg, state])
    send_click.then(
        bot,
        [state, api_key, topk, model, temperature],
        [chat, sources],
        show_progress="minimal",
    ).then(lambda h: h, chat, state)

    msg_submit = msg.submit(add_user, [msg, state], [msg, state])
    msg_submit.then(
        bot,
        [state, api_key, topk, model, temperature],
        [chat, sources],
        show_progress="minimal",
    ).then(lambda h: h, chat, state)


    demo.load(_warmup, inputs=None, outputs=status)


if __name__ == "__main__":
    demo.queue().launch()