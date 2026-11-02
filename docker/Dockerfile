# syntax=docker/dockerfile:1.7
FROM node:24-bookworm-slim@sha256:c86ecead9e55a5210ac1727de38a4ca840bf03dcaf34dc9749addbf383b1920c AS web
WORKDIR /web
COPY gui/web/package*.json ./
RUN npm ci
COPY gui/web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN useradd --uid 10001 --create-home studio && mkdir /data && chown studio /data
WORKDIR /app

FROM base AS dependencies
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY gui/server/ ./gui/server/
RUN pip install --no-cache-dir . ./gui/server

FROM base AS cpu
ENV PATH="/opt/venv/bin:$PATH" NS_WEB_ROOT=/app/web NS_DATA_DIR=/data
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=web /web/dist /app/web
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD ["uvicorn", "studio.app:app", "--host", "0.0.0.0", "--port", "8000"]

# The GPU override selects the CUDA wheel index for the dependency stage.
FROM cpu AS gpu

FROM cpu AS dev
WORKDIR /workspace
ENV PYTHONPATH=/workspace/src:/workspace/gui/server
CMD ["uvicorn", "studio.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/workspace/src", "--reload-dir", "/workspace/gui/server/studio"]
