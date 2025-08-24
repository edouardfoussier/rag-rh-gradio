import os, time
from dotenv import load_dotenv

# Load environment variables BEFORE importing rag modules
load_dotenv(override=True)

import gradio as gr
from rag.retrieval import search, embed
from rag.synth import synth_answer_stream, render_sources
from helpers import linkify_text_with_sources

missing = []
if not os.getenv("HF_API_TOKEN"): missing.append("HF_API_TOKEN (embeddings)")
if not os.getenv("LLM_MODEL"):    print("[INFO] LLM_MODEL not set, using default", flush=True)
print("[ENV] Missing:", ", ".join(missing) or "None", flush=True)
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# def sanity():
#     ok = bool(os.getenv("HF_API_TOKEN"))
#     v = embed("hello world")
#     return f"Token set? {ok}\nEmbedding dim: {len(v)}"

# def rag_chat(user_question, openai_key):
#     if not openai_key:
#         return "❌ Please provide your OpenAI API key."

#     # Inject the key into environment so synth can use it
#     os.environ["OPENAI_API_KEY"] = openai_key

#     # Step 1: Retrieve top passages
#     hits = search(user_question, top_k=8)

#     if not hits:
#         return "❌ Sorry, no relevant information found."

    # # Step 2: Generate synthesized answer
    # try:
    #     final_answer = synth_answer(user_question, hits[:5])
    #     final_answer = linkify(final_answer, hits[:5])
    #     final_answer += "\n\n---\n" + render_sources(hits[:5])
    # except Exception as e:
    #     final_answer = f"❌ Error during synthesis: {e}"

    # return final_answer
# def rag_chat(user_question, openai_key):
#     if not openai_key:
#         yield "❌ Please provide your OpenAI API key."
#         return

#     os.environ["OPENAI_API_KEY"] = openai_key

#     hits = search(user_question, top_k=8)
#     if not hits:
#         yield "❌ Sorry, no relevant information found."
#         return

#     acc = ""
#     try:
#         for piece in synth_answer_stream(user_question, hits[:5]):
#             acc += piece or ""
#             # stream raw text while typing (no links yet to avoid jumpiness)
#             yield acc
#     except Exception as e:
#         partial = acc if acc.strip() else ""
#         yield (partial + ("\n\n" if partial else "") + f"❌ Streaming error: {e}")
#         return

#     final_md = linkify_text_with_sources(acc, hits[:5])
#     yield final_md



# with gr.Blocks() as demo:
#     gr.Markdown("## 🤖 HR Assistant (RAG)\nAsk your question below:")

#     with gr.Row():
#         api_key = gr.Textbox(label="🔑 Your OpenAI API Key", type="password")
    
#     question = gr.Textbox(label="❓ Your Question", placeholder="e.g., Quels sont les droits à congés ?")
    
#     answer = gr.Markdown(label="💡 Assistant Answer")

#     submit_btn = gr.Button("Ask")

#     submit_btn.click(fn=rag_chat, inputs=[question, api_key], outputs=answer)


# if __name__ == "__main__":
#     demo.launch()


def rag_chat(user_question: str, openai_key: str):
    """Generator: streams draft text to a Textbox, then yields final Markdown."""
    if not openai_key:
        yield "❌ Please provide your OpenAI API key.", None
        return

    os.environ["OPENAI_API_KEY"] = openai_key.strip()

    # Step 1: retrieve
    yield "⏳ Recherche des passages pertinents…", None
    hits = search(user_question, top_k=8)
    if not hits:
        yield "❌ Sorry, no relevant information found.", None
        return

    # Step 2: stream LLM synthesis
    acc = ""
    try:
        for piece in synth_answer_stream(user_question, hits[:5]):
            acc += piece or ""
            # Stream into the draft textbox; keep markdown empty during typing
            yield acc, None
    except Exception as e:
        yield f"❌ Error during synthesis: {e}", None
        return

    # Step 3: finalize + linkify citations in Markdown block
    md = linkify_text_with_sources(acc, hits[:5])
    yield acc, md

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 HR Assistant (RAG)\nAsk your question below:")

    with gr.Row():
        api_key = gr.Textbox(label="🔑 Your OpenAI API Key", type="password", placeholder="sk-…")
    question = gr.Textbox(label="❓ Your Question", placeholder="e.g., Quels sont les droits à congés ?")

    # live streaming target
    draft_answer = gr.Markdown(label="💬 Réponse")
    # final pretty markdown with clickable links
    # final_answer = gr.Markdown()

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    submit_btn.click(
        fn=rag_chat,
        inputs=[question, api_key],
        outputs=[draft_answer, final_answer],
        show_progress="full",  # shows loader on the button
    )
    clear_btn.click(lambda: ("", ""), outputs=[draft_answer, final_answer])

if __name__ == "__main__":
    demo.queue().launch()