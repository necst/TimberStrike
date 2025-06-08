FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=1.8.2 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1

# Usa mirror alternativo per evitare 403
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.kumi.systems/ubuntu/|g' /etc/apt/sources.list && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget build-essential git ca-certificates \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    libgmp-dev libntl-dev unzip software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && python -m pip install --upgrade pip setuptools

# Installa le dipendenze necessarie per compilare glibc
RUN apt-get update && apt-get install -y \
    build-essential curl wget unzip gawk bison

# Upgrade glibc to 2.38
RUN mkdir -p /glibc && \
    cd /glibc && \
    curl -O http://ftp.gnu.org/gnu/libc/glibc-2.38.tar.gz && \
    tar -xvzf glibc-2.38.tar.gz && \
    mkdir glibc-build && cd glibc-build && \
    ../glibc-2.38/configure --prefix=/opt/glibc-2.38 && \
    make -j"$(nproc)" && make install

# Set environment to use new glibc
# ENV LD_LIBRARY_PATH=/opt/glibc-2.38/lib

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml ./

# Use system python 3.10 for poetry
RUN poetry env use /usr/bin/python3.10 && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-dev --no-root

# -------- Final Stage --------
FROM base AS final

WORKDIR /app
COPY . .

RUN mkdir -p /app/results && chmod -R 777 /app/results

ENV PYTHONPATH=/app
ENV DEFAULT_SCRIPT=experiments/run_experiment.sh

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash", "-c", "[ -x \"$1\" ] && exec \"$1\" || exec $DEFAULT_SCRIPT", "_", "$0"]
