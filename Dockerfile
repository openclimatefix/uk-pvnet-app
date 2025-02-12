# --- Use conda to install required binaries into venv --- #
FROM quay.io/condaforge/miniforge3:latest AS build-venv

RUN apt-get update && \
    apt-get install git -y && \
    echo "Creating virtualenv at /app/.venv" && \
    conda create --quiet --yes -p /app/.venv python=3.12 esmpy gdal

# --- Build dependencies in own layer --- #
FROM python:3.12 AS build-deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=build-venv /app/.venv /app/.venv

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

# Install only requirements
RUN uv sync --no-dev --no-install-project --compile-bytecode --inexact

FROM build-deps AS build-app

COPY pvnet_app /app/pvnet_app
COPY scripts /app/scripts
# is this data just for testing?
COPY data /app/data

RUN uv sync --no-editable --no-dev --compile-bytecode --inexact

FROM python:3.12-slim

COPY --from=build-app /app/.venv /app/.venv

# This is just a check to make sure it works, we've had problems with this in the past
ENV PATH="/app/.venv/bin:${PATH}"
RUN /app/.venv/bin/python -c "import torchvision"

ENTRYPOINT ["/app/.venv/bin/python/pvnet"]
