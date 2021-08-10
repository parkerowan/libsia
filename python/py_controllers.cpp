/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_controllers.h"
#include "sia/controllers/ilqr.h"
#include "sia/controllers/lqr.h"
#include "sia/controllers/mppi.h"

// Define module
void export_py_controllers(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::CostFunction, PyCostFunction>(m, "CostFunction")
      .def("c", &sia::CostFunction::c, py::arg("x"), py::arg("u"), py::arg("i"))
      .def("cf", &sia::CostFunction::cf, py::arg("x"))
      .def("eval", &sia::CostFunction::eval, py::arg("x"), py::arg("u"));

  py::class_<sia::Controller, PyController>(m, "Controller")
      .def("policy", &sia::Controller::policy, py::arg("state"));

  py::class_<sia::DifferentiableCost, sia::CostFunction, PyDifferentiableCost>(
      m, "DifferentiableCost")
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
      .def("policy", &sia::LQR::policy, py::arg("state"));

  auto ilqr =
      py::class_<sia::iLQR, sia::Controller>(m, "iLQR")
          .def(py::init<sia::LinearizableDynamics&, sia::DifferentiableCost&,
                        const std::vector<Eigen::VectorXd>&, std::size_t,
                        std::size_t, double, double, double, double>(),
               py::arg("dynamics"), py::arg("cost"), py::arg("u0"),
               py::arg("max_iter") = 1, py::arg("max_backsteps") = 1,
               py::arg("epsilon") = 1e-1, py::arg("tau") = 0.5,
               py::arg("min_z") = 1e-1, py::arg("mu") = 0)
          .def("policy", &sia::iLQR::policy, py::arg("state"))
          .def("getControls", &sia::iLQR::getControls,
               py::return_value_policy::reference_internal)
          .def("getStates", &sia::iLQR::getStates,
               py::return_value_policy::reference_internal)
          .def("getMetrics", &sia::iLQR::getMetrics,
               py::return_value_policy::reference_internal);

  py::class_<sia::iLQR::Metrics>(ilqr, "Metrics")
      .def(py::init<>())
      .def_readwrite("iter", &sia::iLQR::Metrics::iter)
      .def_readwrite("dJ", &sia::iLQR::Metrics::dJ)
      .def_readwrite("J", &sia::iLQR::Metrics::J)
      .def_readwrite("z", &sia::iLQR::Metrics::z)
      .def_readwrite("elapsed_us", &sia::iLQR::Metrics::elapsed_us)
      .def_readwrite("backstep_iter", &sia::iLQR::Metrics::backstep_iter)
      .def_readwrite("alpha", &sia::iLQR::Metrics::alpha);

  py::class_<sia::MPPI, sia::Controller>(m, "MPPI")
      .def(py::init<sia::DynamicsModel&, sia::CostFunction&,
                    const std::vector<Eigen::VectorXd>&, std::size_t,
                    const Eigen::MatrixXd&, double>(),
           py::arg("dynamics"), py::arg("cost"), py::arg("u0"),
           py::arg("num_samples"), py::arg("sigma"), py::arg("lam") = 1.0)
      .def("policy", &sia::MPPI::policy, py::arg("state"));
}