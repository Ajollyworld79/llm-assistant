FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal — onnxruntime needs libgomp for OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm fastembed ONNX model into the image (avoids runtime download).
# Multilingual MiniLM (~120MB) supports Danish via ~50 languages.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend/ ./backend/

# Persistent Qdrant volume gets seeded from any local data shipped with the repo
COPY local_qdrant_db/ ./qdrant_data/

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
