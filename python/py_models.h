/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/models/linear_gaussian.h"
#include "sia/models/models.h"
#include "sia/models/nonlinear_gaussian.h"
#include "sia/models/simulate.h"

namespace py = pybind11;

/// DynamicsModel trampoline class
class PyDynamicsModel : public sia::DynamicsModel {
 public:
  // Inherit the constructors
  using sia::DynamicsModel::DynamicsModel;

  // Trampoline (need one for each virtual function)
  sia::Distribution& dynamics(const Eigen::VectorXd& state,
                              const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::DynamicsModel, dynamics,
                           state, control);
  }
};

/// MeasurementModel trampoline class
class PyMeasurementModel : public sia::MeasurementModel {
 public:
  // Inherit the constructors
  using sia::MeasurementModel::MeasurementModel;

  // Trampoline (need one for each virtual function)
  sia::Distribution& measurement(const Eigen::VectorXd& state) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::MeasurementModel,
                           measurement, state);
  }
};

/// LinearizableDynamics trampoline class
class PyLinearizableDynamics : public sia::LinearizableDynamics {
 public:
  // Inherit the constructors
  using sia::LinearizableDynamics::LinearizableDynamics;

  // Trampoline (need one for each virtual function)
  sia::Distribution& dynamics(const Eigen::VectorXd& state,
                              const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::LinearizableDynamics,
                           dynamics, state, control);
  }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd f(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::LinearizableDynamics, f, state,
                           control);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd Q(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::LinearizableDynamics, Q, state,
                           control);
  }
};

/// LinearizableMeasurement trampoline class
class PyLinearizableMeasurement : public sia::LinearizableMeasurement {
 public:
  // Inherit the constructors
  using sia::LinearizableMeasurement::LinearizableMeasurement;

  // Trampoline (need one for each virtual function)
  sia::Distribution& measurement(const Eigen::VectorXd& state) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::LinearizableMeasurement,
                           measurement, state);
  }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd h(const Eigen::VectorXd& state) override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::LinearizableMeasurement, h,
                           state);
  }

  // Trampoline (need one for each virtual function)
  Eigen::MatrixXd R(const Eigen::VectorXd& state) override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, sia::LinearizableMeasurement, R,
                           state);
  }
};

// Define module
void export_py_models(py::module& m_sup);
