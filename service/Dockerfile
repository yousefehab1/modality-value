# Local/Cloud Run demo image for the PTB-XL tabular+waveform fusion service.
# Text/LoRA is intentionally excluded -- see service/app.py's module docstring
# and README "What was cut".
#
# PTB-XL (~1.7GB) is NOT baked into this image; mount it as a volume instead
# of rebuilding the image every time the dataset changes:
#   docker build -t modality-value-service -f service/Dockerfile .
#   docker run -p 8080:8080 -v "$(pwd)/data:/app/data" modality-value-service
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY service ./service
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=src
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PORT=8080

EXPOSE 8080

# Cloud Run sets $PORT; default 8080 matches `make serve` for local parity.
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
