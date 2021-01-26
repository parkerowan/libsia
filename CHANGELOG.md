# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2021-01-25
### Added
- Example notebooks and reamdes for modeling and estimation classes
- Methods for simulating models forward in time
- Methods to vectorize/devectorize distributions
- Buffer class for collecting snapshots of vectors

## [0.1.0] - 2021-01-03
### Added
- C++ library `math` for Runge-Kutta integration and SVD matrix inversion
- C++ library `belief` for distributions, including Gaussian, uniform, weighted particle, and Kernel density (KDE)
- C++ library `models` for Markov process models, including Linear Gaussian, Nonlinear Gaussian, and continuous-time variants
- C++ library `estimators` for recursive Bayesian state estimation, inlcluding KF, EKF, and PF
- C++ library `runner` for estimator and system runner and data recorder
- Logging using glog
- Python bindings using pybind11
- Unit tests using gtest
- Examples for using belief, and lorenz attractor example application
- Dockerfile for dependencies and scripts to work with container
- README for project and examples
- Continuous integration and code coverage reporting
- Changlog for versioning
