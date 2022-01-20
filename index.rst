.. libSIA documentation master file, created by
   sphinx-quickstart on Sat Jul  3 04:18:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to libSIA!
=======================================================================

Sia is a C++/Python library for model-based Reinforcement Learning (or stochastic estimation and optimal control if you prefer). The current scope is on unconstrained continuous state and action vector spaces. Due to the focus on stochastic models, belief and dynamical systems are first-class representations. Algorithms included with the library are general purpose and can be applied to many different applications.

**Features**
   * Finite horizon model predictive control (MPC) including LQR, iLQR, and model predictive path integrals (MPPI).
   * Bayesian estimation including Kalman, extended Kalman, and particle filters.
   * Markov dynamical systems including nonlinear/Gaussian, linear/Gaussian, and their discrete/continuous time variants.
   * Distributions for representing belief including Gaussian, Dirichlet, uniform, categorical, particle, Kernel densities (KDE), Gaussian mixture models (GMM), Gaussian mixture regression (GMR), Gaussian Process Regression (GPR), and Gaussian Process Classification (GPC).
   * Built-in constrained Gradient Descent and Bayesian Optimization.
   * Math functions for Runge-Kutta integration, SVD-based matrix inversion.
   * Minimal dependencies in the core library (Eigen, glog).
   * Python bindings with Pybind11.
   * C++ and Python examples and tutorials.
   * Extensive unit tests with gtest.
   * BSD-3 permissive license.

README
========================================================================

.. include:: README.md
   :parser: myst_parser.sphinx_

Table of Contents
==================================

.. toctree::
   :maxdepth: 1

   docs/notebooks/build.ipynb
   docs/notebooks/overview.ipynb
   docs/tutorials.rst
   examples/examples.rst
   LICENSE.md
   CHANGELOG.md
