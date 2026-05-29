FROM python:3.14-slim

WORKDIR /app

# System deps kept minimal — onnxruntime needs libgomp for OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Qdrant data directory — created empty in the image. On Railway, mount a
# persistent volume here (or set QDRANT_URL for remote Qdrant) so data survives
# redeploys. No DB files are shipped from the repo.
RUN mkdir -p /app/qdrant_data

ENV PYTHONUNBUFFERED=1
ENV PORT=8002
ENV QDRANT_PATH=/app/qdrant_data
ENV DEMO=false
ENV EMBEDDING_PROVIDER=fastembed
ENV EMBEDDING_DIM=384
ENV FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
