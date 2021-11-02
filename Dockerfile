FROM ubuntu:18.04

# -----------------------------------------------------------------------------
# Build dependencies
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    g++ \
    gcovr \
    git \
    lcov \
    valgrind \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Graphics drivers
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && \
    apt install -y \
    ffmpeg \
    mesa-utils \
    ca-certificates \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    nvidia-340 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Library dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y \
    cmake \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    python3-dev \
    python3-pip \
    python3-pytest \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Set python3 to default
RUN update-alternatives \
    --install /usr/bin/python python /usr/bin/python3 10

# pybind11 is required to build the python bindings
RUN cd ~/ \
    && wget -c https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz \
    -O pybind11-v2.6.1.tar.gz \
    && tar -xzvf pybind11-v2.6.1.tar.gz \
    && cd pybind11-2.6.1/ \
    && mkdir build && cd build \
    && cmake .. \
    && make check \
    && make install

# -----------------------------------------------------------------------------
# Test dependencies
# -----------------------------------------------------------------------------
RUN cd /usr/src/gtest/ && \
    cmake . && \
    make && \
    cp *.a /usr/lib

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
RUN python -m pip install --upgrade pip

# Node, npm, pandoc for jupyter
RUN apt-get update && \
    apt-get install -y \
    nodejs \
    npm \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install python programs
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Finalize the container
# -----------------------------------------------------------------------------

# Export the linker library
ENV LD_LIBRARY_PATH /usr/local/lib
ENV PYTHONPATH /usr/local/lib/python3/dist-packages:/libsia/lib

# set working directory
RUN mkdir /libsia
WORKDIR "/libsia"
