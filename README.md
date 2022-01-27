# libSIA - Model-based Reinforcement Learning

[![pipeline status](https://gitlab.com/parkerowan/libsia/badges/master/pipeline.svg)](https://gitlab.com/parkerowan/libsia/commits/master)
[![codecov](https://codecov.io/gl/parkerowan/libsia/branch/master/graph/badge.svg?token=H5P0UCFFR1)](https://codecov.io/gl/parkerowan/libsia)

Sia is a C++/Python library for model-based Reinforcement Learning (or stochastic estimation and optimal control if you prefer). The current scope is on unconstrained continuous state and action vector spaces. Due to the focus on stochastic models, belief and dynamical systems are first-class representations. Algorithms included with the library are general purpose and can be applied to many different applications.

## Features
- Finite horizon model predictive control (MPC) including LQR, iLQR, and model predictive path integrals (MPPI).
- Bayesian estimation including Kalman, extended Kalman, and particle filters.
- Markov dynamical systems including nonlinear/Gaussian, linear/Gaussian, and their discrete/continuous time variants.
- Distributions for representing belief including Gaussian, Dirichlet, uniform, categorical, particle, Kernel densities (KDE), Gaussian mixture models (GMM), Gaussian mixture regression (GMR), Gaussian Process Regression (GPR), and Gaussian Process Classification (GPC).
- Built-in constrained Gradient Descent and Bayesian Optimization.
- Math functions for Runge-Kutta integration, SVD-based matrix inversion.
- Minimal dependencies in the core library (Eigen, glog).
- Python bindings with Pybind11.
- C++ and Python examples and tutorials.
- Extensive unit tests with gtest.
- BSD-3 permissive license.

## Gallery

![Chaotic Lorenz attractor particle filter estimation](./examples/lorenz/lorenz.gif)
![Celestial navigation with iLQR](./examples/navigator/navigator.gif)
![Underactuated cartpole control with iLQR, MPPI](./examples/cartpole/cartpole-ilqr.gif)

## Build
- Install [Docker](https://www.docker.com/).  Check the included Dockerfile for project dependencies.
- Build the Docker container using `scripts/run --build`.
- Launch the Docker container using `scripts/run --bash`.
- Build and install the C++ package with cmake and make.
```bash
mkdir build && cd build
cmake ..
make && make install
make test
```
- Build and install the Python bindings with pip (C++ library must be installed first in previous step).  In the root directory, run
```bash
pip install --upgrade .
```

## Documentation
- Documentation is provided with Sphinx, and is built to `docs/_build/html/index.html`.  It can be built with
```bash
cd build
make docs
```
- Tutorials and in-depth usage are provided in `docs/notebooks`.

## Examples
- C++ examples are built by default with Cmake to `bin`.  Supplemental jupyter notebooks and source are found in `examples`.
