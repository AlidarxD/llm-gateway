FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY gateway ./gateway
COPY migrations ./migrations
COPY config ./config
COPY alembic.ini ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" PROMETHEUS_MULTIPROC_DIR=/tmp/prom
EXPOSE 8000
CMD ["sh", "-c", "rm -rf /tmp/prom && mkdir -p /tmp/prom && alembic upgrade head && uvicorn gateway.app:create_app --factory --host 0.0.0.0 --port 8000 --workers ${GATEWAY_WORKERS:-1} --no-access-log"]
