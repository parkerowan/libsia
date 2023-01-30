/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_controllers.h"
#include "sia/controllers/ilqr.h"
#include "sia/controllers/lqr.h"
#include "sia/controllers/mppi.h"

// Define module
void export_py_controllers(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Controller, PyController>(m, "Controller")
      .def(py::init<>())
      .def("policy", &sia::Controller::policy, py::arg("state"))
      .def("controls", &sia::Controller::controls,
           py::return_value_policy::reference_internal)
      .def("states", &sia::Controller::states,
           py::return_value_policy::reference_internal);

  py::class_<sia::CostFunction, PyCostFunction>(m, "CostFunction")
      .def(py::init<>())
      .def("c", &sia::CostFunction::c, py::arg("x"), py::arg("u"), py::arg("i"))
      .def("cf", &sia::CostFunction::cf, py::arg("x"))
      .def("eval", &sia::CostFunction::eval, py::arg("x"), py::arg("u"));

  py::class_<sia::DifferentiableCost, sia::CostFunction, PyDifferentiableCost>(
      m, "DifferentiableCost")
      .def(py::init<>())
      .def("c", &sia::DifferentiableCost::c, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cf", &sia::DifferentiableCost::cf, py::arg("x"))
      .def("eval", &sia::DifferentiableCost::eval, py::arg("x"), py::arg("u"))
      .def("cx", &sia::DifferentiableCost::cx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cu", &sia::DifferentiableCost::cu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cxx", &sia::DifferentiableCost::cxx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cux", &sia::DifferentiableCost::cux, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cuu", &sia::DifferentiableCost::cuu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cfx", &sia::DifferentiableCost::cfx, py::arg("x"))
      .def("cfxx", &sia::DifferentiableCost::cfxx, py::arg("x"));

  py::class_<sia::QuadraticCost, sia::DifferentiableCost, sia::CostFunction>(
      m, "QuadraticCost")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&>(),
           py::arg("Qf"), py::arg("Q"), py::arg("R"))
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::VectorXd&>(),
           py::arg("Qf"), py::arg("Q"), py::arg("R"), py::arg("xd"))
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&,
                    const std::vector<Eigen::VectorXd>&>(),
           py::arg("Qf"), py::arg("Q"), py::arg("R"), py::arg("xd"))
      .def("setTrajectory", &sia::QuadraticCost::setTrajectory, py::arg("xd"))
      .def("c", &sia::QuadraticCost::c, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cf", &sia::QuadraticCost::cf, py::arg("x"))
      .def("eval", &sia::QuadraticCost::eval, py::arg("x"), py::arg("u"))
      .def("xd", &sia::QuadraticCost::xd, py::arg("i"))
      .def("Qf", &sia::QuadraticCost::Qf)
      .def("Q", &sia::QuadraticCost::Q)
      .def("R", &sia::QuadraticCost::R)
      .def("cx", &sia::QuadraticCost::cx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cu", &sia::QuadraticCost::cu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cxx", &sia::QuadraticCost::cxx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cux", &sia::QuadraticCost::cux, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cuu", &sia::QuadraticCost::cuu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cfx", &sia::QuadraticCost::cfx, py::arg("x"))
      .def("cfxx", &sia::QuadraticCost::cfxx, py::arg("x"));

  py::class_<sia::FunctionalCost, sia::DifferentiableCost, sia::CostFunction>(
      m, "FunctionalCost")
      .def(py::init<sia::TerminalCostFunction, sia::RunningCostFunction>(),
           py::arg("terminal_cost"), py::arg("running_cost"))
      .def("c", &sia::FunctionalCost::c, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cf", &sia::FunctionalCost::cf, py::arg("x"))
      .def("eval", &sia::FunctionalCost::eval, py::arg("x"), py::arg("u"))
      .def("cx", &sia::FunctionalCost::cx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cu", &sia::FunctionalCost::cu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cxx", &sia::FunctionalCost::cxx, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cux", &sia::FunctionalCost::cux, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cuu", &sia::FunctionalCost::cuu, py::arg("x"), py::arg("u"),
           py::arg("i"))
      .def("cfx", &sia::FunctionalCost::cfx, py::arg("x"))
      .def("cfxx", &sia::FunctionalCost::cfxx, py::arg("x"));

  py::class_<sia::LQR, sia::Controller>(m, "LQR")
      .def(py::init<sia::LinearGaussianDynamics&, sia::QuadraticCost&,
                    std::size_t>(),
           py::arg("dynamics"), py::arg("cost"), py::arg("horizon"))
      .def("policy", &sia::LQR::policy, py::arg("state"))
      .def("controls", &sia::LQR::controls,
           py::return_value_policy::reference_internal)
      .def("states", &sia::LQR::states,
           py::return_value_policy::reference_internal)
      .def("feedforward", &sia::LQR::feedforward,
           py::return_value_policy::reference_internal)
      .def("feedback", &sia::LQR::feedback,
           py::return_value_policy::reference_internal);

  auto ilqr = py::class_<sia::iLQR, sia::Controller>(m, "iLQR");

  py::class_<sia::iLQR::Options>(ilqr, "Options")
      .def(py::init<>())
      .def_readwrite("max_lqr_iter", &sia::iLQR::Options::max_lqr_iter)
      .def_readwrite("cost_tol", &sia::iLQR::Options::cost_tol)
      .def_readwrite("max_regularization_iter",
                     &sia::iLQR::Options::max_regularization_iter)
      .def_readwrite("regularization_init",
                     &sia::iLQR::Options::regularization_init)
      .def_readwrite("regularization_min",
                     &sia::iLQR::Options::regularization_min)
      .def_readwrite("regularization_rate",
                     &sia::iLQR::Options::regularization_rate)
      .def_readwrite("max_linesearch_iter",
                     &sia::iLQR::Options::max_linesearch_iter)
      .def_readwrite("linesearch_rate", &sia::iLQR::Options::linesearch_rate)
      .def_readwrite("linesearch_tol_lb",
                     &sia::iLQR::Options::linesearch_tol_lb)
      .def_readwrite("linesearch_tol_ub",
                     &sia::iLQR::Options::linesearch_tol_ub);

  py::class_<sia::iLQR::Metrics>(ilqr, "Metrics")
      .def(py::init<>())
      .def_readwrite("elapsed_us", &sia::iLQR::Metrics::elapsed_us)
      .def_readwrite("lqr_iter", &sia::iLQR::Metrics::lqr_iter)
      .def_readwrite("rho", &sia::iLQR::Metrics::rho)
      .def_readwrite("dJ", &sia::iLQR::Metrics::dJ)
      .def_readwrite("z", &sia::iLQR::Metrics::z)
      .def_readwrite("alpha", &sia::iLQR::Metrics::alpha)
      .def_readwrite("J", &sia::iLQR::Metrics::J);

  ilqr.def(py::init<sia::LinearizableDynamics&, sia::DifferentiableCost&,
                    const std::vector<Eigen::VectorXd>&,
                    const sia::iLQR::Options&>(),
           py::arg("dynamics"), py::arg("cost"), py::arg("u0"),
           py::arg("options") = sia::iLQR::Options())
      .def("policy", &sia::iLQR::policy, py::arg("state"))
      .def("controls", &sia::iLQR::controls,
           py::return_value_policy::reference_internal)
      .def("states", &sia::iLQR::states,
           py::return_value_policy::reference_internal)
      .def("feedforward", &sia::iLQR::feedforward,
           py::return_value_policy::reference_internal)
      .def("feedback", &sia::iLQR::feedback,
           py::return_value_policy::reference_internal)
      .def("metrics", &sia::iLQR::metrics,
           py::return_value_policy::reference_internal);

  auto mppi = py::class_<sia::MPPI, sia::Controller>(m, "MPPI");

  py::class_<sia::MPPI::Options>(mppi, "Options")
      .def(py::init<>())
      .def_readwrite("num_samples", &sia::MPPI::Options::num_samples)
      .def_readwrite("temperature", &sia::MPPI::Options::temperature);

  py::class_<sia::MPPI, sia::Controller>(m, "MPPI")
      .def(py::init<sia::DynamicsModel&, sia::CostFunction&,
                    const std::vector<Eigen::VectorXd>&, const Eigen::MatrixXd&,
                    const sia::MPPI::Options&>(),
           py::arg("dynamics"), py::arg("cost"), py::arg("u0"),
           py::arg("sample_covariance"),
           py::arg("options") = sia::MPPI::Options())
      .def("policy", &sia::MPPI::policy, py::arg("state"))
      .def("controls", &sia::MPPI::controls,
           py::return_value_policy::reference_internal)
      .def("states", &sia::MPPI::states,
           py::return_value_policy::reference_internal)
      .def("rolloutStates", &sia::MPPI::rolloutStates,
           py::return_value_policy::reference_internal)
      .def("rolloutWeights", &sia::MPPI::rolloutWeights,
           py::return_value_policy::reference_internal);
}
