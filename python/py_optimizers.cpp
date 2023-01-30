/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_optimizers.h"

// Define module
void export_py_optimizers(py::module& m_sup) {
  py::module m = m_sup;

  auto gd = py::class_<sia::GradientDescent>(m, "GradientDescent");

  py::class_<sia::GradientDescent::Options>(gd, "Options")
      .def(py::init<>())
      .def_readwrite("n_starts", &sia::GradientDescent::Options::n_starts)
      .def_readwrite("max_iter", &sia::GradientDescent::Options::max_iter)
      .def_readwrite("tol", &sia::GradientDescent::Options::tol)
      .def_readwrite("eta", &sia::GradientDescent::Options::eta)
      .def_readwrite("delta", &sia::GradientDescent::Options::delta);

  gd.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                  const sia::GradientDescent::Options&>(),
         py::arg("lower"), py::arg("upper"),
         py::arg("options") = sia::GradientDescent::Options())
      .def("dimension", &sia::GradientDescent::dimension)
      .def("lower", &sia::GradientDescent::lower)
      .def("upper", &sia::GradientDescent::upper)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GradientDescent::*)(
               sia::GradientDescent::Cost, const Eigen::VectorXd&,
               sia::GradientDescent::Jacobian) const>(
               &sia::GradientDescent::minimize),
           py::arg("f"), py::arg("x0"), py::arg("jacobian") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GradientDescent::*)(
               sia::GradientDescent::Cost, sia::GradientDescent::Jacobian)>(
               &sia::GradientDescent::minimize),
           py::arg("f"), py::arg("jacobian") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GradientDescent::*)(
               sia::GradientDescent::Cost,
               const std::vector<Eigen::VectorXd>& x0,
               sia::GradientDescent::Jacobian) const>(
               &sia::GradientDescent::minimize),
           py::arg("f"), py::arg("x0"), py::arg("jacobian") = nullptr);

  auto bo = py::class_<sia::BayesianOptimizer>(m, "BayesianOptimizer");

  py::enum_<sia::BayesianOptimizer::AcquisitionType>(bo, "AcquisitionType")
      .value("PROBABILITY_IMPROVEMENT",
             sia::BayesianOptimizer::AcquisitionType::PROBABILITY_IMPROVEMENT)
      .value("EXPECTED_IMPROVEMENT",
             sia::BayesianOptimizer::AcquisitionType::EXPECTED_IMPROVEMENT)
      .value("UPPER_CONFIDENCE_BOUND",
             sia::BayesianOptimizer::AcquisitionType::UPPER_CONFIDENCE_BOUND)
      .export_values();

  py::class_<sia::BayesianOptimizer::Options>(bo, "Options")
      .def(py::init<>())
      .def_readwrite("acquisition",
                     &sia::BayesianOptimizer::Options::acquisition)
      .def_readwrite("beta", &sia::BayesianOptimizer::Options::beta)
      .def_readwrite("gradient_descent",
                     &sia::BayesianOptimizer::Options::gradient_descent);

  bo.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, sia::Kernel&,
                  std::size_t, const sia::BayesianOptimizer::Options&>(),
         py::arg("lower"), py::arg("upper"), py::arg("kernel"),
         py::arg("cond_inputs_dim") = 0,
         py::arg("options") = sia::BayesianOptimizer::Options())
      .def("selectNextSample", &sia::BayesianOptimizer::selectNextSample,
           py::arg("u") = Eigen::VectorXd{})
      .def("addDataPoint", &sia::BayesianOptimizer::addDataPoint, py::arg("x"),
           py::arg("y"), py::arg("u") = Eigen::VectorXd{})
      .def("updateModel", &sia::BayesianOptimizer::updateModel,
           py::arg("train") = false)
      .def("getSolution", &sia::BayesianOptimizer::getSolution,
           py::arg("u") = Eigen::VectorXd{})
      .def("objective", &sia::BayesianOptimizer::objective, py::arg("x"),
           py::arg("u") = Eigen::VectorXd{})
      .def("acquisition",
           static_cast<double (sia::BayesianOptimizer::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::BayesianOptimizer::acquisition),
           py::arg("x"), py::arg("u") = Eigen::VectorXd{})
      .def(
          "acquisition",
          static_cast<double (sia::BayesianOptimizer::*)(
              const Eigen::VectorXd&, double,
              sia::BayesianOptimizer::AcquisitionType, const Eigen::VectorXd&)>(
              &sia::BayesianOptimizer::acquisition),
          py::arg("x"), py::arg("target"), py::arg("type"),
          py::arg("u") = Eigen::VectorXd{});
}
