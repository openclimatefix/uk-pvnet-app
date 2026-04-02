.PHONY: create-env sync test all

VENV_DIR := .venv
PYTHON_VERSION := 3.12
CONDA_DEPS := esmpy gdal hdf5=1.14.4

# Create the conda virtual environment with required system packages
$(VENV_DIR):
	conda create --quiet --yes -p $(VENV_DIR) python=$(PYTHON_VERSION) $(CONDA_DEPS)

create-env: $(VENV_DIR)

# Sync Python dependencies into the conda environment using uv
sync: $(VENV_DIR)
	conda run -p $(VENV_DIR) uv sync --inexact

# Run the test suite
test: $(VENV_DIR)
	conda run -p $(VENV_DIR) uv run python -m pytest tests

# Full setup and test in one step
all: create-env sync test
