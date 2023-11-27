/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/optimizers/bo.h"
#include "sia/optimizers/cmaes.h"
#include "sia/optimizers/gd.h"
#include "sia/optimizers/optimizers.h"

namespace py = pybind11;

/// Optimizer trampoline class
class PyOptimizer : public sia::Optimizer {
 public:
  // Inherit the constructors
  using sia::Optimizer::Optimizer;

  // Trampoline (need one for each virtual function)
  void reset() override { PYBIND11_OVERRIDE(void, sia::Optimizer, reset); }

  // Trampoline (need one for each virtual function)
  Eigen::VectorXd step(Optimizer::Cost f,
                       const Eigen::VectorXd& x0,
                       Optimizer::Gradient gradient = nullptr) override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, sia::Optimizer, step, f, x0,
                           gradient);
  }
};

// Define module
void export_py_optimizers(py::module& m_sup);
