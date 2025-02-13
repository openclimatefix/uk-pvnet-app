# --- Use conda to install required binaries into venv --- #
FROM quay.io/condaforge/miniforge3:latest AS build-venv

RUN apt-get update && \
    echo "Creating virtualenv at /app/.venv" && \
    conda create --quiet --yes -p /app/.venv python=3.12 esmpy gdal


# --- Build dependencies in own layer --- #
FROM python:3.12 AS build-deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=build-venv /app/.venv /app/.venv

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

# Install only requirements
RUN mkdir src && uv sync --no-dev --no-install-project --compile-bytecode --inexact

FROM build-deps AS build-app

COPY src /app/src
RUN uv sync --no-editable --no-dev --compile-bytecode --inexact

FROM python:3.12-slim

COPY --from=build-app /app/.venv /app/.venv
COPY --from=build-app /usr/bin/git /usr/bin/git

# This is just a check to make sure it works, we've had problems with this in the past
ENV PATH="/app/.venv/bin:${PATH}"
RUN /app/.venv/bin/python -c "import torchvision"

ENTRYPOINT ["/app/.venv/bin/pvnet-app"]
