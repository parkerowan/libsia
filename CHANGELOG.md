# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2023-11-21
### Added
- Adds a deterministic distribution
### Changed
- Fixes nans in GMR for out of distribution test points
- Fixes a bug in the CMA-ES constructor
- Exposes MPPI costs and controls rollouts 

## [0.4.1] - 2023-04-29
### Added
- Adds Covariance Matrix adaptation (CMA) algorithm
- Adds a damping term for SVD inverse operations
### Changed
- Fixes a potential segfault issue with logging interface
- Fixes a header install path CMake issue

## [0.4.0] - 2023-01-30
### Added
- Adds metrics for controllers and estimators
- Custom logging interface
- Access to control gain terms for LQR/iLQR
- Regularization schedule to iLQR for robust Hessian computation
- Explicit dimensions to dynamics and measurement models
- GPR kernel classes for more transparency into hyperparameters
- KDE smoothing kernel classes for more transparency into bandwidths
- GPR support for evaluating prior
- BayesianOptimizer support for conditioning on input vectors
- GPR edge case test coverage
### Changed
- Moved algorithm parameters to options structs
- Removed GLOG dependency
- Refactored iLQR code for readability
- Changed iLQR and MPPI parameter names to be readable
- Improved numerical stability of iLQR algorithm
- Fixed several bugs in LQR related to tracking formulation
- Fixed a bug in GPR that resulted in singular covariances when evaluating at training points
- Improved GPR/GPC hyperparameter training time and control over search space
- Removed all uses of dynamic memory allocation
- Updated python docs to show usage

## [0.3.3] - 2022-08-03
### Changes
- CMake install bug fixes
- Strongly type enum classes to avoid conflict with other libraries

## [0.3.2] - 2022-04-23
### Added
- CMake install scripts
### Changes
- Documentation

## [0.3.1] - 2022-04-09
### Added
- Deploy stage for Sphinx docs

## [0.3.0] - 2022-01-03
### Added
- Gaussian process classification (GPC)
- Bounded gradient descent optimization
- Methods for training GPR and GPC hyperparameters using gradient descent
- GPR-based Bayesian Optimization
- Categorical and Dirichlet distributions
- Sphinx documentation
- Algorithm benchmarking example
### Changed
- Removed runner class
- Bug fixes to controller algorithms
- Small changes to API for consistency in method naming
- Raw pointers have been changed to shared pointers
- Updated python belief doc to show GPR usage
- Updated README

## [0.2.3] - 2021-08-16
### Added
- Gaussian process regression (GPR) to belief library
- Runtime exceptions for pathological error cases such as failed matrix inversion, cholesky decomposition, dimension mismatch
### Changed
- Parameter set functions now throw exception rather than return false on bad inputs
- Updated python belief doc to show GPR usage
- Updated README

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
