/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/belief/dirichlet.h"
#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gmm.h"
#include "sia/belief/gmr.h"
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

/// Kernel trampoline class
class PyKernel : public sia::Kernel {
 public:
  // Inherit the constructors
  using sia::Kernel::Kernel;

  // Trampoline (need one for each virtual function)
  double evaluate(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::Kernel, evaluate, x);
  }

  // Trampoline (need one for each virtual function)
  sia::Kernel::Type type() const override {
    PYBIND11_OVERRIDE_PURE(sia::Kernel::Type, sia::Kernel, type);
  }
};

// Define module
void export_py_belief(py::module& m_sup);
