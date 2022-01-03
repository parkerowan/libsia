/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/belief/distribution.h"
#include "sia/controllers/controllers.h"
#include "sia/controllers/cost.h"

namespace py = pybind11;

/// Cost function trampoline class
class PyCostFunction : public sia::CostFunction {
 public:
  // Inherit the constructors
  using sia::CostFunction::CostFunction;

  // Trampoline (need one for each virtual function)
  double c(const Eigen::VectorXd& x,
           const Eigen::VectorXd& u,
           std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::CostFunction, c, x, u, i);
  }

  // Trampoline (need one for each virtual function)
  double cf(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::CostFunction, cf, x);
  }
};

/// Differentiable cost function trampoline class
class PyDifferentiableCost : public sia::DifferentiableCost {
 public:
  // Inherit the constructors
  using sia::DifferentiableCost::DifferentiableCost;

  // Trampoline (need one for each virtual function)
  double c(const Eigen::VectorXd& x,
           const Eigen::VectorXd& u,
           std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::DifferentiableCost, c, x, u, i);
  }

  // Trampoline (need one for each virtual function)
  double cf(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(double, sia::DifferentiableCost, cf, x);
  }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd cx(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::DifferentiableCost, cx, x, u,
                           i);
  }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd cu(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::DifferentiableCost, cu, x, u,
                           i);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd cxx(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::DifferentiableCost, cxx, x, u,
                           i);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd cux(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::DifferentiableCost, cux, x, u,
                           i);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd cuu(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::DifferentiableCost, cuu, x, u,
                           i);
  }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd cfx(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::DifferentiableCost, cfx, x);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd cfxx(const Eigen::VectorXd& x) const override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::DifferentiableCost, cfxx, x);
  }
};

/// Controller trampoline class
class PyController : public sia::Controller {
 public:
  // Inherit the constructors
  using sia::Controller::Controller;

  // Trampoline (need one for each virtual function)
  const Eigen::VectorXd& policy(const sia::Distribution& state) override {
    PYBIND11_OVERRIDE_PURE(const Eigen::VectorXd&, sia::Controller, policy,
                           state);
  }
};

// Define module
void export_py_controllers(py::module& m_sup);
