---
license: mit
title: RAG RH (Gradio)
sdk: gradio
emoji: 💻
colorFrom: blue
colorTo: indigo
app_file: app.py
pinned: false
---

# RAG RH (Gradio)

- Embeddings via **HF Inference API** (`feature-extraction`) with `HF_EMBEDDINGS_MODEL` (default `BAAI/bge-m3`)
- Datasets:
  - `edouardfoussier/travail-emploi-clean`
  - `edouardfoussier/service-public-filtered`

## Space Variables

Set in **Settings → Variables**:

- `HF_API_TOKEN` (Write token) — required
- Optional:
  - `HF_EMBEDDINGS_MODEL` (default `BAAI/bge-m3`)
  - `EMBED_COL` (default `embeddings_bge-m3`)
  - `MAX_ROWS_PER_DATASET` (e.g., `2000` to cap memory during testing)