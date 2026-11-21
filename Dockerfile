FROM python:3.14-slim

# Runs as root on purpose: the Railway volume mounted at /app/qdrant_data is
# root-owned, and a non-root user could not write to it.

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Requirements first so this layer is cached until dependencies change
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/

# Qdrant data directory, created empty in the image. On Railway, mount a
# persistent volume here (or set QDRANT_URL for remote Qdrant) so data survives
# redeploys. No DB files are shipped from the repo.
RUN mkdir -p /app/qdrant_data

# Defaults only; Railway variables override these
ENV PYTHONUNBUFFERED=1
ENV PORT=8002
ENV QDRANT_PATH=/app/qdrant_data
ENV DEMO=false
ENV EMBEDDING_PROVIDER=gemini
ENV EMBEDDING_DIM=768

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8002') + '/livez')"

# Qdrant local mode allows a single process: keep --workers 1
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
