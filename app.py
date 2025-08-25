import os
import gradio as gr
from gradio import update as gr_update  # tiny alias
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv(override=True)

from rag.retrieval import search, ensure_ready
from rag.synth import synth_answer_stream
from helpers import _extract_cited_indices, linkify_text_with_sources, _group_sources_md


# ---------- Warm-Up ----------

def _warmup():
    try:
        ensure_ready()
        return "✅ Modèles initialisés !"
    except Exception as e:
        return f"⚠️ Warmup a échoué : {e}"


# ---------- Chat step 1: add user message ----------
def add_user(user_msg: str, history: list[tuple]) -> tuple[str, list[tuple]]:
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "", history
    # append a placeholder assistant turn for streaming
    history = history + [(user_msg, "")]
    return "", history


# ---------- Chat step 2: stream assistant answer ----------
def bot(history: list[tuple], api_key: str, top_k: int):
    """
    Yields (history, sources_markdown) while streaming.
    """
    if not history:
        yield history, "### Sources\n_(none)_"
        return

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key.strip()

    user_msg, _ = history[-1]

    # Retrieval
    k = int(max(top_k, 1))
    try:
        hits = search(user_msg, top_k=k)
    except Exception as e:
        history[-1] = (user_msg, f"❌ Retrieval error: {e}")
        yield history, "### Sources\n_(none)_"
        return

    sources_md = sources_markdown(hits[:k])

    # show a small “thinking” placeholder immediately
    history[-1] = (user_msg, "⏳ Synthèse en cours…")
    yield history, "### 📚 Sources"

    # Streaming LLM
    acc = ""
    try:
        for chunk in synth_answer_stream(user_msg, hits[:k]):
            acc += chunk or ""
            step_hist = deepcopy(history)
            step_hist[-1] = (user_msg, acc)
            yield step_hist, "### 📚 Sources"
    except Exception as e:
        history[-1] = (user_msg, f"❌ Synthèse: {e}")
        yield history, sources_md
        return

    # Finalize + linkify citations
    acc_linked = linkify_text_with_sources(acc, hits[:k])
    history[-1] = (user_msg, acc_linked)
    
    # Construit la section sources à partir des citations réelles [n]
    used = _extract_cited_indices(acc_linked, k)
    grouped_sources = _group_sources_md(hits[:k], used)

    yield history, grouped_sources
    # yield history, sources_md


# ---------- UI ----------
with gr.Blocks(theme="soft", fill_height=True) as demo:
    gr.Markdown("# 🇫🇷 Assistant RH — Chat RAG")
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
        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K passages")
        # you can wire this later; not used now
        save_history = gr.Checkbox(label="Ajouter un modèle eranker")

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
        [state, api_key, topk],
        [chat, sources],
        show_progress="full",
    ).then(lambda h: h, chat, state)

    msg_submit = msg.submit(add_user, [msg, state], [msg, state])
    msg_submit.then(
        bot,
        [state, api_key, topk],
        [chat, sources],
        show_progress="full",
    ).then(lambda h: h, chat, state)


    demo.load(_warmup, inputs=None, outputs=status)


if __name__ == "__main__":
    demo.queue().launch()