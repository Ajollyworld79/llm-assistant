Backend (Quart) — Demo mode

Endpoints

- GET /health — health status
- POST /upload — upload a document (form file)
- GET /documents — list uploaded docs
- POST /search — search (JSON {"query": "...", "top_k": 5})
- POST /reset — clear in-memory store

Notes

- This is a demo scaffold: parsing for PDF (pdfplumber) and DOCX (python-docx) is included; embeddings are deterministic hash-based vectors for demo purposes.
- You can extend parsing and replace the demo embedding with FastEmbed + Qdrant integration later.
- Run with uvicorn: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.

Note: Docker support removed as requested — run services locally.

Qdrant integration

Set the following environment variables in `backend/.env` (or your environment):

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
EMBEDDING_PROVIDER=fastembed  # or sentence-transformers
EMBEDDING_DIM=64
DEMO=false  # set to false to enable Qdrant indexing (requires Qdrant URL)
ADMIN_TOKEN=changeme  # set a strong token to protect admin endpoints

Optional: bring up local Qdrant with Docker (if you want): `docker compose -f backend/docker-compose.yml up` (this is optional; Docker isn't required).

