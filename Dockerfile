# RAG-101 API image — CPU-only inference.
#
# This is the same setup you'd do in a fresh terminal, scripted:
#   1. get Python 3.12
#   2. install dependencies
#   3. copy the code
#   4. run uvicorn
#
# CPU-only torch keeps the image ~2.5 GB instead of ~8 GB: on Linux the
# default PyPI torch wheel bundles CUDA, so we install torch from the CPU
# wheel index FIRST — then sentence-transformers sees it already satisfied
# and never pulls the CUDA build. main.py's device auto-detection finds no
# GPU and falls back to CPU on its own.

FROM python:3.12-slim

WORKDIR /app

# uv (fast installer) — copied from its official image, no curl needed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Layer 1: torch CPU build (the big, rarely-changing layer)
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch==2.7.1

# Layer 2: everything else from pyproject — cached until dependencies change
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

# Layer 3: application code — the only layer that rebuilds on code edits
COPY app/ ./app/

EXPOSE 8000

# Same command as local dev, minus --reload (containers rebuild, not reload)
# and with --host 0.0.0.0 so the API is reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
