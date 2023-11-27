/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_optimizers.h"

// Define module
void export_py_optimizers(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Optimizer, PyOptimizer>(m, "Optimizer")
      .def(py::init<std::size_t, double, std::size_t>(), py::arg("dimension"),
           py::arg("ftol"), py::arg("max_iter"))
      .def("dimension", &sia::Optimizer::dimension)
      .def("reset", &sia::Optimizer::reset)
      .def("step", &sia::Optimizer::step, py::arg("f"), py::arg("x0"),
           py::arg("gradient") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::Optimizer::*)(
               sia::Optimizer::Cost, const Eigen::VectorXd&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::Optimizer::*)(
               sia::Optimizer::Cost, const std::vector<Eigen::VectorXd>&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr);

  auto gd = py::class_<sia::GD, sia::Optimizer>(m, "GD");

  py::class_<sia::GD::Options>(gd, "Options")
      .def(py::init<>())
      .def_readwrite("max_iter", &sia::GD::Options::max_iter)
      .def_readwrite("ftol", &sia::GD::Options::ftol)
      .def_readwrite("eta", &sia::GD::Options::eta)
      .def_readwrite("delta", &sia::GD::Options::delta);

  gd.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                  const sia::GD::Options&>(),
         py::arg("lower"), py::arg("upper"),
         py::arg("options") = sia::GD::Options())
      .def("lower", &sia::GD::lower)
      .def("upper", &sia::GD::upper)
      .def("dimension", &sia::GD::dimension)
      .def("reset", &sia::GD::reset)
      .def("step", &sia::GD::step, py::arg("f"), py::arg("x0"),
           py::arg("gradient") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GD::*)(
               sia::Optimizer::Cost, const Eigen::VectorXd&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::GD::*)(
               sia::Optimizer::Cost, const std::vector<Eigen::VectorXd>&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr);

  auto bo = py::class_<sia::BO, sia::Optimizer>(m, "BO");

  py::enum_<sia::BO::AcquisitionType>(bo, "AcquisitionType")
      .value("PROBABILITY_IMPROVEMENT",
             sia::BO::AcquisitionType::PROBABILITY_IMPROVEMENT)
      .value("EXPECTED_IMPROVEMENT",
             sia::BO::AcquisitionType::EXPECTED_IMPROVEMENT)
      .value("UPPER_CONFIDENCE_BOUND",
             sia::BO::AcquisitionType::UPPER_CONFIDENCE_BOUND)
      .export_values();

  py::class_<sia::BO::Options>(bo, "Options")
      .def(py::init<>())
      .def_readwrite("max_iter", &sia::BO::Options::max_iter)
      .def_readwrite("ftol", &sia::BO::Options::ftol)
      .def_readwrite("acquisition", &sia::BO::Options::acquisition)
      .def_readwrite("beta", &sia::BO::Options::beta)
      .def_readwrite("n_starts", &sia::BO::Options::ftol)
      .def_readwrite("gradient_descent", &sia::BO::Options::gradient_descent);

  bo.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, sia::Kernel&,
                  std::size_t, const sia::BO::Options&>(),
         py::arg("lower"), py::arg("upper"), py::arg("kernel"),
         py::arg("cond_inputs_dim") = 0,
         py::arg("options") = sia::BO::Options())
      .def("selectNextSample", &sia::BO::selectNextSample,
           py::arg("u") = Eigen::VectorXd{})
      .def("addDataPoint", &sia::BO::addDataPoint, py::arg("x"), py::arg("y"),
           py::arg("u") = Eigen::VectorXd{})
      .def("updateModel", &sia::BO::updateModel, py::arg("train") = false)
      .def("getSolution", &sia::BO::getSolution,
           py::arg("u") = Eigen::VectorXd{})
      .def("objective", &sia::BO::objective, py::arg("x"),
           py::arg("u") = Eigen::VectorXd{})
      .def("acquisition",
           static_cast<double (sia::BO::*)(const Eigen::VectorXd&,
                                           const Eigen::VectorXd&)>(
               &sia::BO::acquisition),
           py::arg("x"), py::arg("u") = Eigen::VectorXd{})
      .def("acquisition",
           static_cast<double (sia::BO::*)(
               const Eigen::VectorXd&, double, sia::BO::AcquisitionType,
               const Eigen::VectorXd&)>(&sia::BO::acquisition),
           py::arg("x"), py::arg("target"), py::arg("type"),
           py::arg("u") = Eigen::VectorXd{})
      .def("dimension", &sia::BO::dimension)
      .def("reset", &sia::BO::reset)
      .def("step", &sia::BO::step, py::arg("f"), py::arg("x0"),
           py::arg("gradient") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::BO::*)(
               sia::Optimizer::Cost, const Eigen::VectorXd&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::BO::*)(
               sia::Optimizer::Cost, const std::vector<Eigen::VectorXd>&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr);

  auto cma = py::class_<sia::CMAES, sia::Optimizer>(m, "CMAES");

  py::class_<sia::CMAES::Options>(cma, "Options")
      .def(py::init<>())
      .def_readwrite("max_iter", &sia::CMAES::Options::max_iter)
      .def_readwrite("ftol", &sia::CMAES::Options::ftol)
      .def_readwrite("n_samples", &sia::CMAES::Options::n_samples)
      .def_readwrite("init_stdev", &sia::CMAES::Options::init_stdev)
      .def_readwrite("max_cov_norm", &sia::CMAES::Options::max_cov_norm);

  cma.def(py::init<std::size_t, const sia::CMAES::Options&>(),
          py::arg("dimension"), py::arg("options") = sia::CMAES::Options())
      .def("getSamples", &sia::CMAES::getSamples)
      .def("dimension", &sia::CMAES::dimension)
      .def("reset", &sia::CMAES::reset)
      .def("step", &sia::CMAES::step, py::arg("f"), py::arg("x0"),
           py::arg("gradient") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::CMAES::*)(
               sia::Optimizer::Cost, const Eigen::VectorXd&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr)
      .def("minimize",
           static_cast<Eigen::VectorXd (sia::CMAES::*)(
               sia::Optimizer::Cost, const std::vector<Eigen::VectorXd>&,
               sia::Optimizer::Gradient, sia::Optimizer::Convergence)>(
               &sia::Optimizer::minimize),
           py::arg("f"), py::arg("x0"), py::arg("gradient") = nullptr,
           py::arg("convergence") = nullptr);
}
