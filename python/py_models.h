/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/models/linear_gaussian.h"
#include "sia/models/linear_gaussian_ct.h"
#include "sia/models/models.h"
#include "sia/models/nonlinear_gaussian.h"
#include "sia/models/nonlinear_gaussian_ct.h"
#include "sia/models/simulate.h"

namespace py = pybind11;

/// MarkovProcess trampoline class
class PyMarkovProcess : public sia::MarkovProcess {
 public:
  // Inherit the constructors
  using sia::MarkovProcess::MarkovProcess;

  // Trampoline (need one for each virtual function)
  sia::Distribution& dynamics(const Eigen::VectorXd& state,
                              const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::MarkovProcess, dynamics,
                           state, control);
  }

  // Trampoline (need one for each virtual function)
  sia::Distribution& measurement(const Eigen::VectorXd& state) override {
    PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::MarkovProcess, measurement,
                           state);
  }
};

// Define module
void export_py_models(py::module& m_sup);
