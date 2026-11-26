# Slim serving image: only core + serve deps (no DuckDB, MLflow, dev tools).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 1) deps layer (cached unless lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --group serve --no-install-project

# 2) app code + config + model bundle
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-default-groups --group serve
COPY models/champion ./models/champion

RUN useradd --create-home app && chown -R app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" MODEL_DIR=/app/models/champion
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "taxi_demand.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
