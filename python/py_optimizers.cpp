/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_optimizers.h"

// Define module
void export_py_optimizers(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::GradientDescent>(m, "GradientDescent")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double,
                    double, double>(),
           py::arg("lower"), py::arg("upper"), py::arg("tol") = 1e-6,
           py::arg("eta") = 0.5, py::arg("delta") = 0.5)
      .def("dimension", &sia::GradientDescent::dimension)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GradientDescent::*)(
               std::function<double(const Eigen::VectorXd&)>,
               const Eigen::VectorXd&) const>(&sia::GradientDescent::minimize),
           py::arg("f"), py::arg("x0"))
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GradientDescent::*)(
               std::function<double(const Eigen::VectorXd&)>,
               const std::vector<Eigen::VectorXd>& x0) const>(
               &sia::GradientDescent::minimize),
           py::arg("f"), py::arg("x0"));

  //   py::class_<sia::BayesianOptimizer>(m, "BayesianOptimizer")
  //       .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&>(),
  //            py::arg("lower"), py::arg("upper"))
  //       .def("selectNextSample", &sia::BayesianOptimizer::selectNextSample)
  //       .def("addDataPoint", &sia::BayesianOptimizer::addDataPoint,
  //       py::arg("x"),
  //            py::arg("y"))
  //       .def("updateModel", &sia::BayesianOptimizer::updateModel)
  //       .def("getSolution", &sia::BayesianOptimizer::getSolution)
  //       .def("gpr", &sia::BayesianOptimizer::gpr);
}
