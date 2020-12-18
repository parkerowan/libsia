FROM ubuntu:18.04

# -----------------------------------------------------------------------------
# Developer tools
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y \
    autotools-dev \
    build-essential \
    clang-format \
    clang-tidy \
    curl \
    g++ \
    gcovr \
    git \
    lcov \
    libbz2-dev \
    libicu-dev \
    nano \
    pkg-config \
    software-properties-common \
    tmux \
    tree \
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
    python-dev \
    python-pip \
    python3-dev \
    python3-pip \
    python3-pytest \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# pybind11 is required to build the python bindings
RUN cd ~/ \
    && wget -c https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz \
    -O pybind11-v2.6.1.tar.gz \
    && tar -xzvf pybind11-v2.6.1.tar.gz \
    && cd pybind11-2.6.1/ \
    && mkdir build && cd build \
    && cmake .. \
    && make check -j4 \
    && make install

# -----------------------------------------------------------------------------
# Test and example dependencies
# -----------------------------------------------------------------------------

# gtest is required for the tests
RUN cd /usr/src/gtest/ && \
    cmake . && \
    make && \
    cp *.a /usr/lib

# cxxopts is required for the examples
RUN cd ~/ \
    && wget https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz \
    && tar -xzvf v2.2.1.tar.gz \
    && cd cxxopts-2.2.1 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

# -----------------------------------------------------------------------------
# Python install
# -----------------------------------------------------------------------------

# Set python3 to default
RUN update-alternatives \
    --install /usr/bin/python python /usr/bin/python3 10

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install python programs
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Finalize the container
# -----------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y ffmpeg

# Export the linker library
ENV LD_LIBRARY_PATH /usr/local/lib:/libsia/lib
ENV PYTHONPATH /libsia/lib

# set working directory
RUN mkdir /libsia
WORKDIR "/libsia"
