"""ASGI entry point: ``uvicorn backend.app.main:app --workers 1`` (local Qdrant needs a single worker)."""

from backend.app import create_app

app = create_app()
