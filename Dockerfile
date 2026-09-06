# Multi-stage: the build stage compiles wheels, the runtime stage carries none
# of the toolchain. Matters here because torch and faiss pull in large build
# dependencies that have no business in a running container.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

# Non-root. The service reads a corpus and serves HTTP; it never needs root,
# and a container that runs as root is one container escape from being one.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Cache the sentence-transformer weights into the image so a cold start does
# not depend on Hugging Face being reachable.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN mkdir -p $HF_HOME && chown -R appuser:appuser /app $HF_HOME
USER appuser

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1 \
    INDEX_DIR=/data/index \
    CORPUS_PATH=/data/oncology_trials.csv

EXPOSE 8000

# Hits the real endpoint rather than just checking the process is alive: a
# service with no index loaded is up but useless, and /health reports that.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
