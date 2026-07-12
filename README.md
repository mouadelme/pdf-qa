---
title: PDF Question Answering
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---
# PDF Question Answering

A retrieval-augmented question-answering application for PDF documents.

## Features

- PDF text extraction and normalization
- Paragraph-aware document chunking
- Sentence Transformer embeddings
- Semantic document retrieval
- FLAN-T5 answer generation
- Retrieved source display
- Browser interface built with Gradio

## Architecture

```text
PDF upload
    ↓
Text extraction and cleaning
    ↓
Paragraph-aware chunking
    ↓
Sentence Transformer embeddings
    ↓
Semantic retrieval
    ↓
FLAN-T5 answer generation
    ↓
Answer and supporting passage

Set `sdk_version` to the version actually installed locally:

```bash
python -c "import gradio; print(gradio.__version__)"