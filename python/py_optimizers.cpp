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

  py::enum_<sia::ObjectiveType>(m, "ObjectiveType")
      .value("GPR_OBJECTIVE", sia::ObjectiveType::GPR_OBJECTIVE)
      .export_values();

  py::enum_<sia::AcquistionType>(m, "AcquistionType")
      .value("PROBABILITY_IMPROVEMENT",
             sia::AcquistionType::PROBABILITY_IMPROVEMENT)
      .value("EXPECTED_IMPROVEMENT", sia::AcquistionType::EXPECTED_IMPROVEMENT)
      .value("UPPER_CONFIDENCE_BOUND",
             sia::AcquistionType::UPPER_CONFIDENCE_BOUND)
      .export_values();

  py::class_<sia::BayesianOptimizer>(m, "BayesianOptimizer")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                    sia::ObjectiveType, sia::AcquistionType, std::size_t>(),
           py::arg("lower"), py::arg("upper"),
           py::arg("objective") = sia::ObjectiveType::GPR_OBJECTIVE,
           py::arg("acquisition") = sia::AcquistionType::EXPECTED_IMPROVEMENT,
           py::arg("nstarts") = 10)
      .def("selectNextSample", &sia::BayesianOptimizer::selectNextSample)
      .def("addDataPoint", &sia::BayesianOptimizer::addDataPoint, py::arg("x"),
           py::arg("y"))
      .def("updateModel", &sia::BayesianOptimizer::updateModel)
      .def("getSolution", &sia::BayesianOptimizer::getSolution)
      .def("optimizer", &sia::BayesianOptimizer::optimizer,
           py::return_value_policy::reference_internal)
      .def("surrogate", &sia::BayesianOptimizer::surrogate,
           py::return_value_policy::reference_internal);

  py::class_<sia::SurrogateModel, PySurrogateModel>(m, "SurrogateModel")
      .def("addDataPoint", &sia::SurrogateModel::addDataPoint, py::arg("x"),
           py::arg("y"))
      .def("initialized", &sia::SurrogateModel::initialized)
      .def("updateModel", &sia::SurrogateModel::updateModel)
      .def("objective", &sia::SurrogateModel::objective, py::arg("x"),
           py::return_value_policy::reference_internal)
      .def("acquisition", &sia::SurrogateModel::acquisition, py::arg("x"),
           py::arg("target"), py::arg("type"));

  py::class_<sia::GPRSurrogateModel, sia::SurrogateModel>(m,
                                                          "GPRSurrogateModel")
      .def(py::init<double, double, double, double>(), py::arg("varn") = 1e-4,
           py::arg("varf") = 1, py::arg("length") = 1, py::arg("beta") = 1)
      .def("addDataPoint", &sia::GPRSurrogateModel::addDataPoint, py::arg("x"),
           py::arg("y"))
      .def("initialized", &sia::GPRSurrogateModel::initialized)
      .def("updateModel", &sia::GPRSurrogateModel::updateModel)
      .def("objective", &sia::GPRSurrogateModel::objective, py::arg("x"),
           py::return_value_policy::reference_internal)
      .def("acquisition", &sia::GPRSurrogateModel::acquisition, py::arg("x"),
           py::arg("target"), py::arg("type"));
}
