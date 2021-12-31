/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
      .def("options", &sia::GradientDescent::options)
      .def("setOptions", &sia::GradientDescent::setOptions, py::arg("options"))
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

  //   py::enum_<sia::ObjectiveType>(m, "ObjectiveType")
  //       .value("GPR_OBJECTIVE", sia::ObjectiveType::GPR_OBJECTIVE)
  //       .export_values();

  //   py::enum_<sia::AcquisitionType>(m, "AcquisitionType")
  //       .value("PROBABILITY_IMPROVEMENT",
  //              sia::AcquisitionType::PROBABILITY_IMPROVEMENT)
  //       .value("EXPECTED_IMPROVEMENT",
  //       sia::AcquisitionType::EXPECTED_IMPROVEMENT)
  //       .value("UPPER_CONFIDENCE_BOUND",
  //              sia::AcquisitionType::UPPER_CONFIDENCE_BOUND)
  //       .export_values();

  //   py::class_<sia::BayesianOptimizer>(m, "BayesianOptimizer")
  //       .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
  //                     sia::ObjectiveType, sia::AcquisitionType,
  //                     std::size_t>(),
  //            py::arg("lower"), py::arg("upper"),
  //            py::arg("objective") = sia::ObjectiveType::GPR_OBJECTIVE,
  //            py::arg("acquisition") =
  //            sia::AcquisitionType::EXPECTED_IMPROVEMENT, py::arg("nstarts") =
  //            10)
  //       .def("selectNextSample", &sia::BayesianOptimizer::selectNextSample)
  //       .def("addDataPoint", &sia::BayesianOptimizer::addDataPoint,
  //       py::arg("x"),
  //            py::arg("y"))
  //       .def("updateModel", &sia::BayesianOptimizer::updateModel)
  //       .def("getSolution", &sia::BayesianOptimizer::getSolution)
  //       .def("optimizer", &sia::BayesianOptimizer::optimizer,
  //            py::return_value_policy::reference_internal)
  //       .def("surrogate", &sia::BayesianOptimizer::surrogate,
  //            py::return_value_policy::reference_internal);
}
