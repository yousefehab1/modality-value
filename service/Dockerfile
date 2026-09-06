# Demo image for the PTB-XL tabular+waveform fusion service. Text/LoRA is
# intentionally excluded -- see service/app.py's module docstring and README
# "What was cut".
#
# The full PTB-XL dataset (~1.7GB) is NOT baked in and is not needed: the
# image bakes in service/artifacts/ instead (fitted models + the small val
# and test-fold derivatives service/app.py needs; see
# scripts/export_service_artifacts.py), so the container is self-contained
# and boots without any volume mount.
#   docker build -t modality-value-service -f service/Dockerfile .
#   docker run -p 7860:7860 modality-value-service
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
# 7860 is Hugging Face Spaces' default app port; Cloud Run overrides $PORT
# itself at runtime, so this default only matters for local/HF use.
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
