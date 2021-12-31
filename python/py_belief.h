/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/belief/categorical.h"
#include "sia/belief/dirichlet.h"
#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gmm.h"
#include "sia/belief/gmr.h"
#include "sia/belief/gpc.h"
#include "sia/belief/gpr.h"
#include "sia/belief/helpers.h"
#include "sia/belief/kernel_density.h"
#include "sia/belief/particles.h"
#include "sia/belief/uniform.h"

namespace py = pybind11;

/// Distribution trampoline class
class PyDistribution : public sia::Distribution {
 public:
  // Inherit the constructors
  using sia::Distribution::Distribution;

  // Trampoline (need one for each virtual function)
  std::size_t dimension() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, sia::Distribution, dimension);
  }

  // Trampoline (need one for each virtual function)
  const Eigen::VectorXd sample() override {
    PYBIND11_OVERRIDE_PURE(const Eigen::VectorXd, sia::Distribution, sample);
  }

  // Trampoline (need one for each virtual function)
  double logProb(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::Distribution, logProb, x);
  }

  // Trampoline (need one for each virtual function)
  const Eigen::VectorXd mean() const override {
    PYBIND11_OVERRIDE_PURE(const Eigen::VectorXd, sia::Distribution, mean);
  }

  // Trampoline (need one for each virtual function)
  const Eigen::VectorXd mode() const override {
    PYBIND11_OVERRIDE_PURE(const Eigen::VectorXd, sia::Distribution, mode);
  }

  // Trampoline (need one for each virtual function)
  const Eigen::MatrixXd covariance() const override {
    PYBIND11_OVERRIDE_PURE(const Eigen::MatrixXd, sia::Distribution,
                           covariance);
  }

  // Trampoline (need one for each virtual function)
  const Eigen::VectorXd vectorize() const override {
    PYBIND11_OVERRIDE_PURE(const Eigen::VectorXd, sia::Distribution, vectorize);
  }

  // Trampoline (need one for each virtual function)
  bool devectorize(const Eigen::VectorXd& data) override {
    PYBIND11_OVERRIDE_PURE(bool, sia::Distribution, devectorize, data);
  }
};

/// Inference trampoline class
class PyInference : public sia::Inference {
 public:
  // Inherit the constructors
  using sia::Inference::Inference;

  // Trampoline (need one for each virtual function)
  const sia::Distribution& predict(const Eigen::VectorXd& x) override {
    PYBIND11_OVERRIDE_PURE(const sia::Distribution&, sia::Inference, predict,
                           x);
  }

  // Trampoline (need one for each virtual function)
  std::size_t inputDimension() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, sia::Inference, inputDimension);
  }

  // Trampoline (need one for each virtual function)
  std::size_t outputDimension() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, sia::Inference, outputDimension);
  }
};

// Define module
void export_py_belief(py::module& m_sup);
