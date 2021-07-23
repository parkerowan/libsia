# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2021-08-09
### Added
- Gaussian mixture models (GMM) and Gaussian mixture regression (GMR) to belief library
### Changed
- Refactored the models library to split dynamics and measurement models into distinct classes
- Updated dependent libraries, examples, and python docs to reflect the refactor
- Fix Gaussian LDLT cholesky covariance decomposition bug
- Fix random number generator seed issue, updates rng to mersene twister
- Updated README

## [0.2.1] - 2021-07-30
### Added
- C++ library `controllers` for model predictive control (MPC), including LQR, iLQR and model predictive path integrals (MPPI)
- Controller examples 'motor', 'navigator', and 'cartpole'
- Numerical derivatives for vector functions
- Sphinx doc folder for tutorials
- Unit tests for added components
### Changed
- Moved 'belief', 'models', 'estimators', and new 'controllers' tutorial notebooks to docs
- Removed support for graphics from the Docker container for security
- Updated README
- Bux fixes

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
