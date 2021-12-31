/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include "sia/optimizers/bayesian_optimizer.h"
#include "sia/optimizers/gradient_descent.h"

namespace py = pybind11;

/// SurrogateModel trampoline class
// class PySurrogateModel : public sia::SurrogateModel {
//  public:
//   // Inherit the constructors
//   using sia::SurrogateModel::SurrogateModel;

//   // Trampoline (need one for each virtual function)
//   bool initialized() const override {
//     PYBIND11_OVERRIDE_PURE(bool, sia::SurrogateModel, initialized);
//   }

//   // Trampoline (need one for each virtual function)
//   void updateModel() override {
//     PYBIND11_OVERRIDE_PURE(void, sia::SurrogateModel, updateModel);
//   }

//   // Trampoline (need one for each virtual function)
//   const sia::Distribution& objective(const Eigen::VectorXd& x) override {
//     PYBIND11_OVERRIDE_PURE(sia::Distribution&, sia::SurrogateModel,
//     objective,
//                            x);
//   }

//   // Trampoline (need one for each virtual function)
//   double acquisition(const Eigen::VectorXd& x,
//                      double target,
//                      sia::AcquisitionType type) override {
//     PYBIND11_OVERRIDE_PURE(double, sia::SurrogateModel, acquisition, x,
//     target,
//                            type);
//   }
// };

// Define module
void export_py_optimizers(py::module& m_sup);
