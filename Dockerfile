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
RUN conda install -c conda-forge xesmf esmpy h5py -y
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Change to app folder
WORKDIR /app

# Install library
RUN pip install -e .
RUN python -c "import torchvision"

RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

CMD ["python", "-u","pvnet_app/app.py"]
