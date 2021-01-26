/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sia/estimators/estimators.h"
#include "sia/estimators/extended_kalman_filter.h"
#include "sia/estimators/kalman_filter.h"
#include "sia/estimators/particle_filter.h"

namespace py = pybind11;

/// MarkovProcess trampoline class
class PyRecursiveBayesEstimator : public sia::RecursiveBayesEstimator {
 public:
  // Inherit the constructors
  using sia::RecursiveBayesEstimator::RecursiveBayesEstimator;

  // Trampoline (need one for each virtual function)
  const sia::Distribution& getBelief() const override {
    PYBIND11_OVERRIDE_PURE(const sia::Distribution&,
                           sia::RecursiveBayesEstimator, getBelief);
  }

  // Trampoline (need one for each virtual function)
  const sia::Distribution& estimate(const Eigen::VectorXd& observation,
                                    const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(const sia::Distribution&,
                           sia::RecursiveBayesEstimator, estimate, observation,
                           control);
  }

  // Trampoline (need one for each virtual function)
  const sia::Distribution& predict(const Eigen::VectorXd& control) override {
    PYBIND11_OVERRIDE_PURE(const sia::Distribution&,
                           sia::RecursiveBayesEstimator, predict, control);
  }

  // Trampoline (need one for each virtual function)
  const sia::Distribution& correct(
      const Eigen::VectorXd& observation) override {
    PYBIND11_OVERRIDE_PURE(const sia::Distribution&,
                           sia::RecursiveBayesEstimator, correct, observation);
  }
};

// Define module
void export_py_estimators(py::module& m_sup);
