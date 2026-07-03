# --- Use conda to install required binaries into venv --- #
FROM quay.io/condaforge/miniforge3:latest AS build-venv

RUN apt-get update && \
    echo "Creating virtualenv at /app/.venv" && \
    conda create --quiet --yes -p /app/.venv python=3.12 "esmf=*=nompi_*" esmpy hdf5=1.14.4


# --- Build dependencies --- #
FROM python:3.12 AS build-deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=build-venv /app/.venv /app/.venv

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

# Install only requirements
RUN mkdir src && uv sync --no-dev --no-install-project --compile-bytecode --inexact

# --- Build the package --- #
FROM build-deps AS build-app

# Install the app
COPY src /app/src
RUN uv sync --no-editable --no-dev --compile-bytecode --inexact

# --- Runtime image --- #
FROM python:3.12-slim

# Copy required elements of the builder image
COPY --from=build-app /app/.venv /app/.venv

# This is just a check to make sure it works, we've had problems with this in the past
ENV PATH="/app/.venv/bin:${PATH}"
RUN /app/.venv/bin/python -c "import torch"

ENTRYPOINT ["/app/.venv/bin/pvnet-app"]
