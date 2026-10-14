# Georgian Document RAG


## What it does

- Extracts text from Georgian PDFs using the native PDF text layer when available
- Falls back to Tesseract OCR for scanned or low-quality pages
- Normalizes, chunks, and stores document content in SQLite
- Embeds content and searches it with Qdrant for semantic retrieval
- Answers user questions in Russian using the retrieved context and page citations like `[стр. 3]`
- Ships with both a CLI workflow and a FastAPI + Jinja2 web UI

## Key features

- Georgian source documents, Russian user-facing answers
- Native text extraction + OCR fallback for PDF ingestion
- Vector search over indexed chunks with Qdrant
- Grounded answer generation with citation validation
- Local-only web interface for document upload, indexing, and Q&A
- OpenRouter and Ollama provider support via a common LLM abstraction

## Stack

- Python 3.12
- FastAPI + Jinja2
- SQLAlchemy + Alembic
- Qdrant
- SentenceTransformers / BAAI/bge-m3
- PyMuPDF + Tesseract OCR
- Ollama / OpenRouter LLM integration

## Use case

This project is designed for knowledge work over Georgian-language documents, especially scanned or mixed-format PDFs where the information needs to be searched and cited reliably. It is optimized for grounded answers rather than unconstrained generation.

## Status

Prototype / research project for local document retrieval and citation-based Q&A.
