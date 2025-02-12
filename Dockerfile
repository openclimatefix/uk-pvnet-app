# Build stage
FROM continuumio/miniconda3 AS builder

SHELL ["/bin/bash", "-l", "-c"]

# Install only the necessary build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    g++ \
    gcc \
    libgeos++-dev \
    libproj-dev \
    proj-data \
    proj-bin && \
    rm -rf /var/lib/apt/lists/*

# Install Python and conda packages first (less frequent changes)
RUN conda install python=3.12 && \
    conda install -c conda-forge xesmf esmpy h5py pytorch-cpu=2.3.1 torchvision -y && \
    conda clean -afy

# Copy only requirements first to leverage cache
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install pip requirements
RUN pip install --no-cache-dir torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy source code (changes more frequently)
COPY setup.py README.md ./
COPY pvnet_app/ ./pvnet_app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install the application
RUN pip install --no-cache-dir -e .

# Verify torch installation
RUN python -c "import torchvision"

# Final stage
FROM continuumio/miniconda3 AS runner

# Copy conda environment from builder (multistaging stage1)
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

WORKDIR /app

# Install testing dependencies if needed (ARG is scoped to each stage)
ARG TESTING=0
RUN if [ "$TESTING" = 1 ]; then pip install --no-cache-dir pytest pytest-cov coverage; fi

CMD ["python", "-u","pvnet_app/app.py"]