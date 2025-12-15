Qdrant Chatbot (Demo Mode)

This repository is a scaffold for a Qdrant-like chatbot demo. It provides:

- A Python Quart backend that simulates FastEmbed/Qdrant search results (demo mode).
- A simple static frontend demonstrating the chat and admin upload flow.

All functionality runs in **demo mode** (no real LLM or Qdrant connection).

Quick start

1. Create a virtual environment and install backend deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. Frontend

- Quick static demo (no npm required): open `frontend/static.html` in a browser or serve it with a simple static server:

  cd frontend
  python3 -m http.server 5174
  # Then open http://127.0.0.1:5174/static.html

  Note: The backend has CORS allowed, so the static page can fetch API endpoints at http://127.0.0.1:8000.

- Full dev React frontend (optional):

   cd frontend
   npm install
   npm run dev

   Open http://127.0.0.1:5173/ in a browser and use the React chat and admin upload.

Project structure

- `backend/` - Quart backend (demo search, upload)
- `frontend/` - Static demo frontend (chat + admin)
- `README.md` - This file

Notes

- This is intentionally a demo scaffold. You can replace the demo search with real Qdrant or embedding services later.
