FROM continuumio/miniconda3

ARG TESTING=0

SHELL ["/bin/bash", "-l", "-c"]

RUN apt-get update && \
    apt-get install git unzip g++ gcc libgeos++-dev libproj-dev proj-data proj-bin -y

# Copy files
COPY setup.py app/setup.py
COPY README.md app/README.md
COPY requirements.txt app/requirements.txt
COPY pvnet_app/ app/pvnet_app/
COPY tests/ app/tests/
COPY scripts/ app/scripts/
COPY data/ app/data/

# Install requirements
RUN conda install python=3.12
RUN conda install -c conda-forge xesmf esmpy h5py pytorch-cpu=2.3.1 torchvision -y
RUN pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cpu

# Change to app folder
WORKDIR /app

# Install library
RUN pip install -e .


# Download models so app can used cached versions instead of pulling from huggingface
RUN python scripts/cache_default_models.py

# This is just a check to make sure it works, we've had problems with this in the past
RUN python -c "import torchvision"


RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

CMD ["python", "-u","pvnet_app/app.py"]
