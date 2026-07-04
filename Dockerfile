# syntax=docker/dockerfile:1
#
# Multi-stage build. The builder installs deps, BAKES the embedding model into the
# Hugging Face cache, and precomputes the retrieval index. The runtime stage copies
# only what it needs and runs fully offline (HF_HUB_OFFLINE=1) as a non-root user.
# This makes container start-up fast, deterministic, and network-independent.

# ----------------------------- builder -----------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBED_CACHE_DIR=/opt/fastembed

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ingest ./ingest
COPY data/kb.json ./data/kb.json

# Downloads bge-small into /opt/fastembed AND writes data/index.npz (baked index).
RUN python -m ingest.ingest --kb data/kb.json --index data/index.npz

# ----------------------------- runtime -----------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBED_CACHE_DIR=/opt/fastembed \
    HF_HUB_OFFLINE=1

WORKDIR /app

# Installed packages + console scripts (uvicorn) from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Baked model cache + precomputed index.
COPY --from=builder /opt/fastembed /opt/fastembed
COPY --from=builder /build/data/index.npz ./data/index.npz

# Application code + data + UI.
COPY app ./app
COPY data/kb.json ./data/kb.json
COPY static ./static

# Run as a non-root numeric user; the model cache must be readable by it.
RUN useradd --uid 10001 --create-home appuser \
    && chown -R 10001:10001 /app /opt/fastembed
USER 10001

EXPOSE 8000

# Liveness/readiness are handled by Kubernetes probes; this HEALTHCHECK helps
# `docker run` users and CI smoke tests.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
