# Georgian Document RAG

RAG prototype for extracting Georgian PDF documents and answering grounded questions in Russian.

## Local setup

1. Install dependencies:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and adjust service URLs or model names.
3. Start Qdrant and Ollama, install the configured Ollama model, and install Tesseract language data for `kat` and `eng`.
4. Apply migrations:

   ```powershell
   .\venv\Scripts\alembic.exe upgrade head
   ```

5. Check all runtime dependencies before ingestion:

   ```powershell
   .\venv\Scripts\python.exe main.py preflight
   ```

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -m "not integration"
.\venv\Scripts\python.exe -m pytest -m integration
```

Changing `EMBEDDING_MODEL` or `EMBEDDING_SIZE` requires a compatible Qdrant collection and re-indexing all documents.
