---
license: mit
title: 🇫🇷 Assistant RH — RAG Chatbot
sdk: gradio
emoji: 📚
colorFrom: indigo
colorTo: purple
app_file: app.py
pinned: true
short_description: 👉 RAG-powered AI assistant for French Human Resources
tags:
- gradio
- rag
- faiss
- openai
- hr
- human-resources
- law
- france
- french
- chatbot
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6668057ef7604601278857f5/JeivLn409aMRCqx6RwO2J.png
---

# 🇫🇷 RAG-powered HR Assistant

👉 **An AI assistant specialised in French Human Resources, powered by Retrieval-Augmented Generation (RAG) and based on official public datasets.**  

[![Hugging Face Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-blue)](https://huggingface.co/spaces/edouardfoussier/rag-rh-assistant)

![App Screenshot](assets/screenshot2.png)

---

## ✨ What is this?

This project is an **AI assistant** for HR topics in the **French labor law and public administration HR practices**.  
It combines **retrieval** over trusted sources with **LLM synthesis**, and cites its sources.

- UI: **Gradio**
- Retrieval: **FAISS** (fallback: NumPy)
- Embeddings: **HF Inference API**
- LLM: **OpenAI** (BYO API Key)

---

## 📚 Datasets & Attribution

This space relies on **public HR datasets** curated by [**AgentPublic**](https://huggingface.co/datasets/AgentPublic):
- [Service-Public dataset](https://huggingface.co/datasets/AgentPublic/service-public)
- [Travail-Emploi dataset](https://huggingface.co/datasets/AgentPublic/travail-emploi)

For this project, I built **cleaned and filtered derivatives** hosted under my profile:
- [edouardfoussier/service-public-filtered](https://huggingface.co/datasets/edouardfoussier/service-public-filtered)
- [edouardfoussier/travail-emploi-clean](https://huggingface.co/datasets/edouardfoussier/travail-emploi-clean)

---

## ⚙️ How it works

1. **Question** → User asks in French (e.g., “DPAE : quelles obligations ?”).  
2. **Retrieve** → FAISS searches semantic vectors from the datasets.  
3. **Synthesize** → The LLM writes a concise, factual answer with citations `[1], [2], …`.  
4. **Explain** → The “Sources” panel shows the original articles used for answer generation

---

## 🔑 BYOK

The app never stores your OpenAI key; it’s used in-session only.

---

## 🧩 Configuration notes

- FAISS is used when available; otherwise we fall back to NumPy dot-product search.
- The retriever loads vectors from the datasets and keeps a compressed cache at runtime (/tmp/rag_index.npz) to speed up cold starts.
- You can change the Top-K slider in the UI; it controls both retrieval and the number of passages given to the LLM.

---

## 🚀 Run locally

### 1) Clone & install
```bash
git clone https://huggingface.co/spaces/edouardfoussier/rag-rh-assistant
cd rag-rh-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
Key env vars:
- HF_API_TOKEN → required for embeddings via HF Inference API
- HF_EMBEDDINGS_MODEL → defaults to BAAI/bge-m3
- EMBED_COL → name of the embedding column in the dataset (defaults to embeddings_bge-m3)
- OPENAI_API_KEY → optional at startup (you can also enter it in the UI)
- LLM_MODEL → e.g. gpt-4o-mini (configurable)
- LLM_BASE_URL → default https://api.openai.com/v1

### 3) Launch
```bash
python app.py
```

Open http://127.0.0.1:7860 and enter your OpenAI API key in the sidebar (or set it in .env).

---

## 📊 Roadmap

- Reranking (cross-encoder)
- Multi-turn memory
- More datasets (other ministries, codes)
- Hallucination checks & eval (faithfulness)
- Multi-LLM backends

---

## 🙌 Credits

- Original data: [**AgentPublic**](https://huggingface.co/datasets/AgentPublic)
- Built with: Hugging Face Spaces, Gradio, FAISS, OpenAI